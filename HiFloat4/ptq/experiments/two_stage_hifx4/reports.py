from __future__ import annotations

from typing import Any

from ...state import QuantizedWeightState


def summarize_two_stage_state(state: QuantizedWeightState) -> dict[str, Any]:
    return {
        "algorithm": state.algorithm,
        "format": state.format_name,
        "stage1_enabled": bool(state.metadata.get("two_stage_stage1", False)),
        "stage2_enabled": bool(state.metadata.get("two_stage_stage2", False)),
        "stage1_improvement": state.report.get("stage1_improvement"),
        "stage2_improvement": state.report.get("stage2_improvement"),
        "stage1_scale_change_rate": state.report.get("stage1_scale_change_rate"),
        "stage2_scale_change_rate": state.report.get("stage2_scale_change_rate"),
    }
