"""将 HiF4 adapter 接入现有 Wan2.2 生成器的最小 hooks。"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

from .hif4_backend import get_hif4_stats, hif4_backend_metadata, reset_hif4_stats
from .hif4_linear import replace_linear_with_hif4


def add_hif4_arguments(parser: argparse.ArgumentParser) -> None:
    """给已有 generator parser 增加 HiF4 参数，并把 ``hif4`` 加入 ``--mode``。"""

    for action in parser._actions:
        if "--mode" in action.option_strings and action.choices is not None:
            choices = list(action.choices)
            if "hif4" not in choices:
                choices.append("hif4")
                action.choices = choices
            break

    parser.add_argument("--hif4_main_repo", type=str, default="", help="HiF4_Sp repo root containing HiFloat4")
    parser.add_argument("--hif4_include_regex", type=str, default="", help="Optional regex selecting Linear modules")
    parser.add_argument("--hif4_exclude_regex", type=str, default="", help="Optional regex excluding Linear modules")
    parser.add_argument("--hif4_no_activation_qdq", action="store_true", default=False, help="Disable online activation QDQ (W4-only)")
    parser.add_argument("--hif4_force_fp32", action="store_true", default=True, help="Call official QDQ with force_fp32=True")
    parser.add_argument("--hif4_dry_run", action="store_true", default=False, help="Only list selected modules; no weight QDQ")
    parser.add_argument("--hif4_setup_only", action="store_true", default=False, help="Replace modules/write setup report, then exit")
    parser.add_argument("--hif4_report_json", type=str, default="", help="Setup report JSON")
    parser.add_argument("--hif4_runtime_report_json", type=str, default="", help="Runtime report JSON")


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def setup_hif4_if_needed(
    pipe: Any,
    args: argparse.Namespace,
    logger: logging.Logger | None = None,
) -> dict[str, Any] | None:
    """在 WanI2V 构造后替换 low/high noise DiT 的目标 Linear。"""

    if getattr(args, "mode", None) != "hif4":
        return None

    logger = logger or logging.getLogger(__name__)
    hif4_repo = getattr(args, "hif4_main_repo", "") or os.environ.get("HIF4_MAIN_REPO", "")
    if hif4_repo:
        os.environ["HIF4_MAIN_REPO"] = str(Path(hif4_repo).expanduser().resolve())

    reset_hif4_stats()
    reports: dict[str, Any] = {}
    for label in ("low_noise_model", "high_noise_model"):
        model = getattr(pipe, label, None)
        if model is None:
            reports[label] = {"present": False}
            continue

        logger.info("[%s] move to %s for one-time HiF4 weight QDQ", label, pipe.device)
        model.to(pipe.device).eval().requires_grad_(False)
        report = replace_linear_with_hif4(
            model,
            include_regex=getattr(args, "hif4_include_regex", "") or None,
            exclude_regex=getattr(args, "hif4_exclude_regex", "") or None,
            activation_qdq=not bool(getattr(args, "hif4_no_activation_qdq", False)),
            force_fp32=bool(getattr(args, "hif4_force_fp32", True)),
            dry_run=bool(getattr(args, "hif4_dry_run", False) and not getattr(args, "hif4_setup_only", False)),
            quant_device=str(pipe.device),
        )
        reports[label] = {"present": True, **report}
        logger.info(
            "[%s] HiF4 selected=%d skipped=%d dry_run=%s",
            label,
            report["selected_count"],
            report["skipped_count"],
            report["dry_run"],
        )

        if bool(getattr(args, "hif4_setup_only", False)):
            import torch

            model.cpu()
            torch.cuda.empty_cache()

    payload = {
        "mode": "hif4",
        "definition": "BF16 checkpoint -> one-time HiF4 weight RTN-QDQ -> online HiF4 activation QDQ -> BF16 F.linear",
        "metadata": hif4_backend_metadata(),
        "stats_after_setup": get_hif4_stats(),
        "reports": reports,
    }
    output_dir = Path(getattr(args, "output_dir", "outputs/vbench_hif4"))
    report_path = getattr(args, "hif4_report_json", "") or output_dir / "hif4_setup_report.json"
    _write_json(report_path, payload)
    logger.info("HiF4 setup report written to %s", report_path)

    if bool(getattr(args, "hif4_dry_run", False) or getattr(args, "hif4_setup_only", False)):
        logger.info("HIF4_SETUP_ONLY_DONE" if getattr(args, "hif4_setup_only", False) else "HIF4_DRY_RUN_DONE")
        raise SystemExit(0)
    return payload


def write_hif4_runtime_report(
    args: argparse.Namespace,
    *,
    generated: int,
    skipped: int,
    failed: int,
    manifest: str | Path,
    logger: logging.Logger | None = None,
) -> dict[str, Any] | None:
    """在生成循环结束后写入完整 QDQ 调用统计。"""

    if getattr(args, "mode", None) != "hif4":
        return None
    logger = logger or logging.getLogger(__name__)
    output_dir = Path(getattr(args, "output_dir", "outputs/vbench_hif4"))
    report_path = getattr(args, "hif4_runtime_report_json", "") or output_dir / "hif4_runtime_report.json"
    payload = {
        "mode": "hif4",
        "definition": "HiF4 W4A4 RTN-QDQ fake-quant runtime report",
        "metadata": hif4_backend_metadata(),
        "stats_after_generation": get_hif4_stats(),
        "generated": int(generated),
        "skipped": int(skipped),
        "failed": int(failed),
        "manifest": str(manifest),
        "output_dir": str(output_dir),
    }
    _write_json(report_path, payload)
    logger.info("HiF4 runtime report written to %s", report_path)
    return payload
