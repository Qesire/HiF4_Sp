#!/usr/bin/env python3
from __future__ import annotations

from hifx4_source_bootstrap import bootstrap_source_tree

SOURCE_TREE_PATHS = bootstrap_source_tree(__file__)

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from HiFloat4.ptq.adapters.wan_i2v import native_hif4_weight_qdq
from HiFloat4.ptq.algorithms.gptq_original import OriginalGPTQConfig, quantize_gptq_original
from HiFloat4.ptq.formats.hifx4 import HIFX4Format
from HiFloat4.ptq.state import CurvatureStats

EXPERTS = ("low_noise_model", "high_noise_model")


def load_dict(path: Path) -> dict[str, Any]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise TypeError(f"expected dict: {path}")
    return obj


def save_atomic(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def curvature_from_record(name: str, rec: dict[str, Any], expected_domain: str) -> CurvatureStats:
    matrix = rec.get("hessian", rec.get("matrix"))
    if not torch.is_tensor(matrix):
        raise TypeError(f"missing Hessian for {name}")
    out = CurvatureStats(
        matrix=matrix.detach().float().cpu().contiguous(),
        nsamples=int(rec["nsamples"]),
        in_features=int(rec.get("in_features", matrix.shape[0])),
        normalization=str(rec.get("normalization", "mean_xtx")),
        domain=str(rec.get("domain", "")),
        module_name=name,
    )
    out.validate()
    if out.domain != expected_domain:
        raise ValueError(f"{name}: expected curvature domain={expected_domain}, got {out.domain}")
    return out


def empty_cache_payload(
    *, expert: str, block: int, variant: str, source: Path, quantizer: str,
    activation_qdq: bool, curvature_domain: str, hmeta: dict[str, Any],
) -> dict[str, Any]:
    return {
        "__meta__": {
            "artifact": "wan_hif4_preprocessed_linear_cache_shard_v48",
            "schema_version": 48,
            "expert": expert,
            "block": block,
            "variant": variant,
            "source_prepared_shard": str(source),
            "format": "hifx4",
            "weight_quantizer": quantizer,
            "activation_qdq": activation_qdq,
            "curvature_domain": curvature_domain,
            "runtime_activation_domain": "hif4_a4",
            "curvature_runtime_domain_match": False,
            "experiment_design": "noqdq_hessian_w4a4_runtime",
            "manifest_sha256": hmeta.get("manifest_sha256"),
            "ordered_sample_ids_sha256": hmeta.get("ordered_sample_ids_sha256"),
            "resolved_image_paths_sha256": hmeta.get("resolved_image_paths_sha256"),
        },
        "low_noise_model": {}, "high_noise_model": {},
    }


def cache_record(
    *, name: str, weight_qdq: torch.Tensor, smooth_scale: torch.Tensor,
    activation_qdq: bool, algorithm: str, transforms: list[str],
    metadata: dict[str, Any], report: dict[str, Any],
) -> dict[str, Any]:
    out_features, in_features = map(int, weight_qdq.shape)
    return {
        "in_features": in_features,
        "out_features": out_features,
        "weight_qdq": weight_qdq.detach().cpu().contiguous(),
        "smooth_scale": smooth_scale.detach().float().cpu().contiguous(),
        "activation_qdq": bool(activation_qdq),
        "force_fp32": True,
        "algorithm": algorithm,
        "transforms": transforms,
        "source_name": name,
        "metadata": metadata,
        "report": report,
    }


def weighted_error(weight: torch.Tensor, qweight: torch.Tensor, hessian: torch.Tensor) -> float:
    error = weight.detach().float().cpu() - qweight.detach().float().cpu()
    h = hessian.detach().float().cpu()
    value = torch.sum((error @ h) * error)
    return float(value.item())


def main() -> None:
    ap = argparse.ArgumentParser(description="Quantize one prepared Wan block to paired RTN/GPTQ cache shards (v4.8).")
    ap.add_argument("--prepared-shard", required=True)
    ap.add_argument("--hessian-shard", required=True)
    ap.add_argument("--rtn-output", required=True)
    ap.add_argument("--gptq-output", required=True)
    ap.add_argument("--rtn-variant", required=True)
    ap.add_argument("--gptq-variant", required=True)
    ap.add_argument("--quant-device", default="cuda:0")
    ap.add_argument("--gptq-block-size", type=int, default=64)
    ap.add_argument("--gptq-damp-percent", type=float, default=0.01)
    ap.add_argument("--gptq-compute-device", default="cuda:0")
    ap.add_argument("--expected-curvature-domain", choices=["smooth_bf16_noqdq"], required=True)
    ap.add_argument("--activation-qdq", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--allow-noqdq-hessian-w4a4-runtime", action="store_true")
    args = ap.parse_args()

    if not args.activation_qdq:
        raise ValueError("current experiment requires HiF4 W4A4 runtime")
    if not args.allow_noqdq_hessian_w4a4_runtime:
        raise ValueError("explicit --allow-noqdq-hessian-w4a4-runtime is required")

    prepared_path = Path(args.prepared_shard).resolve()
    hessian_path = Path(args.hessian_shard).resolve()
    prepared = load_dict(prepared_path)
    hessian = load_dict(hessian_path)
    pmeta = prepared.get("__meta__", {})
    hmeta = hessian.get("__meta__", {})
    expert = str(pmeta.get("expert", ""))
    block = int(pmeta.get("block", -1))
    if expert not in EXPERTS or block < 0:
        raise ValueError(f"invalid prepared shard metadata: {pmeta}")
    if str(hmeta.get("expert", "")) != expert or int(hmeta.get("block", -1)) != block:
        raise ValueError("prepared/curvature shard expert-block mismatch")
    if str(hmeta.get("domain", "")) != args.expected_curvature_domain:
        raise ValueError(
            f"curvature shard mismatch: expected {args.expected_curvature_domain}, got {hmeta.get('domain')}"
        )
    prepared_domain = pmeta.get("curvature_domain")
    if prepared_domain is not None and str(prepared_domain) != args.expected_curvature_domain:
        raise ValueError(
            f"prepared weight was built with {prepared_domain}, cannot quantize against {args.expected_curvature_domain}"
        )

    records = prepared.get(expert, {})
    hrecords = hessian.get(expert, {})
    if set(records) != set(hrecords):
        raise RuntimeError(f"module mismatch prepared={sorted(records)} hessian={sorted(hrecords)}")

    rtn_payload = empty_cache_payload(
        expert=expert, block=block, variant=args.rtn_variant, source=prepared_path,
        quantizer="rtn", activation_qdq=bool(args.activation_qdq),
        curvature_domain=args.expected_curvature_domain, hmeta=hmeta,
    )
    gptq_payload = empty_cache_payload(
        expert=expert, block=block, variant=args.gptq_variant, source=prepared_path,
        quantizer="gptq", activation_qdq=bool(args.activation_qdq),
        curvature_domain=args.expected_curvature_domain, hmeta=hmeta,
    )
    gptq_cfg = OriginalGPTQConfig(
        block_size=args.gptq_block_size,
        damp_percent=args.gptq_damp_percent,
        compute_device=args.gptq_compute_device,
    )
    gptq_cfg.validate()
    backend = HIFX4Format(force_fp32=True)
    audits: list[dict[str, Any]] = []

    for idx, name in enumerate(sorted(records), start=1):
        rec = records[name]
        weight = rec["weight"].detach()
        scale = rec["scale"].detach()
        transforms = list(rec.get("transforms", ["smoothquant_wan_wgt"]))
        curvature = curvature_from_record(name, hrecords[name], args.expected_curvature_domain)

        rtn_qdq = native_hif4_weight_qdq(weight.to(args.quant_device), force_fp32=True).to(
            device="cpu", dtype=weight.dtype
        )
        qstate = quantize_gptq_original(weight, curvature, backend, gptq_cfg)
        gptq_qdq = qstate.dequantized_weight.to(device="cpu", dtype=weight.dtype)

        hmat = curvature.as_mean()
        rtn_err = weighted_error(weight, rtn_qdq, hmat)
        gptq_err = weighted_error(weight, gptq_qdq, hmat)
        rho = gptq_err / max(rtn_err, 1e-30)
        different_rate = float((rtn_qdq.float() != gptq_qdq.float()).float().mean().item())
        audits.append({
            "name": name,
            "rtn_weighted_error": rtn_err,
            "gptq_weighted_error": gptq_err,
            "rho": rho,
            "gptq_better": bool(rho < 1.0),
            "rtn_gptq_weight_different_rate": different_rate,
        })

        rtn_payload[expert][name] = cache_record(
            name=name, weight_qdq=rtn_qdq, smooth_scale=scale,
            activation_qdq=bool(args.activation_qdq), algorithm="native_hif4_rtn",
            transforms=transforms,
            metadata={
                "format": "hifx4", "weight_quantizer": "rtn",
                "quant_device": args.quant_device,
                "source_preprocess": pmeta.get("preprocess", "smooth"),
                "curvature_domain": args.expected_curvature_domain,
            },
            report={**dict(rec.get("magr_report", {})), **audits[-1]},
        )
        gptq_payload[expert][name] = cache_record(
            name=name, weight_qdq=gptq_qdq, smooth_scale=scale,
            activation_qdq=bool(args.activation_qdq), algorithm="hif4_gptq_original",
            transforms=transforms,
            metadata={
                **qstate.metadata, "format": "hifx4", "weight_quantizer": "gptq",
                "source_preprocess": pmeta.get("preprocess", "smooth"),
                "curvature_domain": args.expected_curvature_domain,
            },
            report={**dict(rec.get("magr_report", {})), **qstate.report, **audits[-1]},
        )
        del rtn_qdq, gptq_qdq, qstate
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[{idx}/{len(records)}] paired quantization {expert}.{name} rho={rho:.6g}")

    rtn_path = Path(args.rtn_output).resolve()
    gptq_path = Path(args.gptq_output).resolve()
    save_atomic(rtn_payload, rtn_path)
    save_atomic(gptq_payload, gptq_path)
    rhos = sorted(float(x["rho"]) for x in audits)
    median_rho = rhos[len(rhos) // 2] if rhos else math.inf
    summary = {
        "artifact": "wan_hif4_pair_audit_v48",
        "expert": expert,
        "block": block,
        "module_count": len(records),
        "curvature_domain": args.expected_curvature_domain,
        "activation_qdq": bool(args.activation_qdq),
        "runtime_activation_domain": "hif4_a4",
        "experiment_design": "noqdq_hessian_w4a4_runtime",
        "rho_fraction_lt_1": (sum(x["gptq_better"] for x in audits) / len(audits)) if audits else 0.0,
        "median_rho": median_rho,
        "median_weight_different_rate": sorted(x["rtn_gptq_weight_different_rate"] for x in audits)[len(audits)//2] if audits else 0.0,
        "modules": audits,
        "rtn_output": str(rtn_path), "rtn_bytes": rtn_path.stat().st_size,
        "gptq_output": str(gptq_path), "gptq_bytes": gptq_path.stat().st_size,
    }
    gptq_path.with_suffix(gptq_path.suffix + ".pair.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "modules"}, indent=2, ensure_ascii=False))
    print("WAN_HIF4_PAIRED_QUANT_SHARD_V48_OK")


if __name__ == "__main__":
    main()
