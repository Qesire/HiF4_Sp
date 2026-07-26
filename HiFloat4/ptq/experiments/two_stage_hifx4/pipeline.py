from __future__ import annotations

import torch

from ...algorithms.gptq_original import quantize_gptq_original
from ...formats.hifx4 import HIFX4Format
from ...state import CurvatureStats, DeviationStats, QuantizedWeightState
from .config import TwoStageHiF4GPTQConfig
from .stage1 import optimize_stage1_metadata
from .stage2 import optimize_stage2_base_scales


@torch.no_grad()
def quantize_gptq_two_stage(
    weight: torch.Tensor,
    curvature: CurvatureStats,
    format_backend: HIFX4Format,
    config: TwoStageHiF4GPTQConfig | None = None,
    *,
    deviation: DeviationStats | None = None,
) -> QuantizedWeightState:
    """Experimental wrapper; the original GPTQ inner loop is never modified."""

    cfg = config or TwoStageHiF4GPTQConfig()
    cfg.validate()
    stage1_result = None
    precomputed = None
    if cfg.stage1.enabled:
        stage1_result = optimize_stage1_metadata(
            weight, curvature, format_backend, cfg.stage1
        )
        precomputed = stage1_result.metadata

    original = quantize_gptq_original(
        weight,
        curvature,
        format_backend,
        cfg.original_gptq,
        precomputed_metadata=precomputed,
    )
    original.metadata["experimental"] = True
    original.metadata["two_stage_stage1"] = bool(cfg.stage1.enabled)
    if stage1_result is not None:
        original.report.update(stage1_result.report)

    if cfg.stage2.enabled:
        result = optimize_stage2_base_scales(
            weight,
            curvature,
            original,
            format_backend,
            cfg.stage2,
            deviation=deviation,
        ).state
    else:
        result = original
        result.algorithm = "gptq_two_stage"
        result.metadata["two_stage_stage2"] = False

    result.algorithm = "gptq_two_stage"
    result.metadata["pipeline_variant"] = "experimental_two_stage_metadata_optimization"
    return result
