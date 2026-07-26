#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import torch

from HiFloat4.ptq.algorithms.gptq_original import OriginalGPTQConfig, quantize_gptq_original
from HiFloat4.ptq.formats.hifx4 import HIFX4Format, HIFX4TensorMetadata
from HiFloat4.ptq.state import CurvatureStats


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def metadata_from_weight(weight: torch.Tensor) -> HIFX4TensorMetadata:
    states = []
    for start in range(0, weight.shape[1], 64):
        states.append(HIFX4Format.encode_group(weight[:, start:start + 64].float(), group_start=start).metadata)
    return HIFX4TensorMetadata.from_group_states(states, in_features=weight.shape[1])


def decode_state(state) -> torch.Tensor:
    record = state.format_state
    if not isinstance(record, dict) or record.get("kind") != "hifx4_quantized_tensor":
        raise AssertionError("missing hifx4 format_state")
    metadata = HIFX4TensorMetadata.from_record(record["metadata"])
    codes = record["codes"].to(torch.int8)
    out = torch.empty(codes.shape, dtype=torch.float32)
    backend = HIFX4Format()
    for group_index in range(metadata.num_groups):
        start = group_index * metadata.group_size
        width = int(metadata.group_widths[group_index])
        group = metadata.group_state(group_index, format_backend=backend, device="cpu")
        out[:, start:start + width] = codes[:, start:start + width].float() / 4.0 * group.full_scale
    return out


def frozen_rtn(weight: torch.Tensor, metadata: HIFX4TensorMetadata) -> tuple[torch.Tensor, torch.Tensor]:
    backend = HIFX4Format()
    q = torch.empty_like(weight, dtype=torch.float32)
    codes = torch.empty_like(weight, dtype=torch.int8)
    for group_index in range(metadata.num_groups):
        state = metadata.group_state(group_index, format_backend=backend, device="cpu")
        start = group_index * metadata.group_size
        for local in range(state.width):
            value, code = backend.quantize_column_with_code(weight[:, start + local].float(), state, local)
            q[:, start + local] = value.float()
            codes[:, start + local] = code.to(torch.int8)
    return q, codes


def independent_gptq_frozen(weight: torch.Tensor, hessian: torch.Tensor, metadata: HIFX4TensorMetadata,
                            block_size: int, damp_percent: float) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Independent, compact GPTQ reference for frozen HiF4 metadata."""
    W = weight.float().clone()
    H = 0.5 * (hessian.float() + hessian.float().T)
    dead = torch.diag(H) == 0
    if dead.any():
        idx = torch.nonzero(dead, as_tuple=False).flatten()
        H[idx, idx] = 1.0
        W[:, idx] = 0.0
    damp = max(float(damp_percent * torch.diag(H).mean()), 1e-8)
    chol = torch.linalg.cholesky(H + torch.eye(H.shape[0]) * damp)
    hinv = torch.linalg.cholesky(torch.cholesky_inverse(chol), upper=True)
    backend = HIFX4Format()
    Q = torch.zeros_like(W)
    codes = torch.zeros_like(W, dtype=torch.int8)
    d = W.shape[1]
    for start in range(0, d, block_size):
        end = min(start + block_size, d)
        work = W[:, start:end].clone()
        err_block = torch.zeros_like(work)
        hblock = hinv[start:end, start:end]
        for local in range(end - start):
            col = start + local
            group_index = col // metadata.group_size
            group = metadata.group_state(group_index, format_backend=backend, device="cpu")
            q, code = backend.quantize_column_with_code(work[:, local], group, col % metadata.group_size)
            q = q.float()
            Q[:, col] = q
            codes[:, col] = code.to(torch.int8)
            err = (work[:, local] - q) / hblock[local, local]
            work[:, local:] -= err[:, None] @ hblock[local, local:][None, :]
            err_block[:, local] = err
        if end < d:
            W[:, end:] -= err_block @ hinv[start:end, end:]
    return Q, codes, damp


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(a.float() - b.float()) /
                 torch.linalg.vector_norm(b.float()).clamp_min(1e-12))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=27018)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    rows, dim = 5, 128
    w = torch.randn(rows, dim) * 0.2
    x = torch.randn(384, dim)
    h = x.T @ x / x.shape[0]
    metadata = metadata_from_weight(w)
    curvature = CurvatureStats(h, x.shape[0], dim, "mean_xtx", "original", "toy")
    cfg = OriginalGPTQConfig(block_size=64, damp_percent=0.01, compute_device="cpu")

    checks: list[dict[str, Any]] = []
    def add(name: str, passed: bool, **detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), **detail})

    # 1. Independent frozen-metadata reference equivalence.
    state = quantize_gptq_original(w, curvature, HIFX4Format(), cfg, precomputed_metadata=metadata)
    q_ref, codes_ref, damp_ref = independent_gptq_frozen(w, h, metadata, 64, 0.01)
    codes_impl = state.format_state["codes"].to(torch.int8)
    add("independent_frozen_reference", torch.equal(codes_impl, codes_ref) and rel(state.dequantized_weight, q_ref) < 1e-7,
        code_equal=torch.equal(codes_impl, codes_ref), relative_q_error=rel(state.dequantized_weight, q_ref), reference_damp=damp_ref)

    # 2. Stored codes+metadata must reconstruct the returned QDQ tensor.
    decoded = decode_state(state)
    add("format_state_roundtrip", rel(decoded, state.dequantized_weight) < 1e-7,
        relative_decode_error=rel(decoded, state.dequantized_weight))

    # 3. Scaling H by a positive scalar should preserve codes because damping is relative.
    curvature_scaled = CurvatureStats(17.0 * h, x.shape[0], dim, "mean_xtx", "original", "toy_scaled")
    scaled = quantize_gptq_original(w, curvature_scaled, HIFX4Format(), cfg, precomputed_metadata=metadata)
    add("positive_hessian_scale_invariance", torch.equal(codes_impl, scaled.format_state["codes"]),
        code_equal=torch.equal(codes_impl, scaled.format_state["codes"]), relative_q_error=rel(state.dequantized_weight, scaled.dequantized_weight))

    # 4. With diagonal H and frozen metadata, no cross-column compensation exists: GPTQ == frozen RTN.
    diag_h = torch.diag(torch.linspace(0.3, 2.0, dim))
    diag_curv = CurvatureStats(diag_h, 1, dim, "mean_xtx", "original", "diag")
    diag_state = quantize_gptq_original(w, diag_curv, HIFX4Format(), cfg, precomputed_metadata=metadata)
    q_rtn, codes_rtn = frozen_rtn(w, metadata)
    add("diagonal_hessian_reduces_to_frozen_rtn", torch.equal(diag_state.format_state["codes"], codes_rtn) and rel(diag_state.dequantized_weight, q_rtn) < 1e-7,
        code_equal=torch.equal(diag_state.format_state["codes"], codes_rtn), relative_q_error=rel(diag_state.dequantized_weight, q_rtn))

    # 5. Reported weighted error must match direct evaluation against the undamped H.
    e = w.float() - state.dequantized_weight.float()
    direct = float(torch.sum((e @ h.float()) * e))
    reported = float(state.report["hessian_weighted_error"])
    add("weighted_error_report", math.isclose(direct, reported, rel_tol=2e-6, abs_tol=1e-6), direct=direct, reported=reported)

    # 6. Dynamic metadata for a one-group tensor must equal metadata generated from the input group.
    w64 = w[:, :64].contiguous()
    h64 = h[:64, :64].contiguous()
    curv64 = CurvatureStats(h64, x.shape[0], 64, "mean_xtx", "original", "one_group")
    dynamic64 = quantize_gptq_original(w64, curv64, HIFX4Format(), cfg)
    expected64 = metadata_from_weight(w64).to_record()
    got64 = dynamic64.format_state["metadata"]
    metadata_equal = all(torch.equal(got64[k], expected64[k]) for k in ["base_scales", "level2_exponents", "level3_exponents", "group_widths"])
    add("one_group_repository_metadata_semantics", metadata_equal, metadata_equal=metadata_equal)

    # 7. Dead-column and damping path must remain finite and deterministic.
    h_dead = h.clone(); h_dead[0, :] = 0; h_dead[:, 0] = 0
    dead_curv = CurvatureStats(h_dead, x.shape[0], dim, "mean_xtx", "original", "dead")
    dead_state = quantize_gptq_original(w, dead_curv, HIFX4Format(), cfg, precomputed_metadata=metadata)
    add("dead_column_path", torch.isfinite(dead_state.dequantized_weight).all() and dead_state.metadata["dead_columns"] == 1,
        dead_columns=dead_state.metadata["dead_columns"], finite=bool(torch.isfinite(dead_state.dequantized_weight).all()))

    passed = all(c["passed"] for c in checks)
    payload = {
        "artifact": "hifx4_gptq_independent_validation_v1",
        "seed": args.seed,
        "torch": torch.__version__,
        "source_hint": os.environ.get("BASE_KIT_ROOT"),
        "passed": passed,
        "checks": checks,
    }
    atomic_json(Path(args.output), payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.strict and not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
