import torch

from HiFloat4.ptq.algorithms.gptq_original import OriginalGPTQConfig
from HiFloat4.ptq.experiments.two_stage_hifx4 import (
    HiF4Stage1Config,
    HiF4Stage2Config,
    TwoStageHiF4GPTQConfig,
    optimize_stage1_metadata,
    quantize_gptq_two_stage,
)
from HiFloat4.ptq.formats.hifx4 import HIFX4Format
from HiFloat4.ptq.state import CurvatureStats


def spd(size: int) -> torch.Tensor:
    torch.manual_seed(5)
    x = torch.randn(size, size)
    return x.T @ x + torch.eye(size) * 0.5


def test_stage1_never_worsens_its_group_objective():
    torch.manual_seed(2)
    weight = torch.randn(4, 64)
    result = optimize_stage1_metadata(
        weight,
        CurvatureStats(spd(64), 16, 64),
        HIFX4Format(),
        HiF4Stage1Config(scale_neighbors=2),
    )
    assert result.report["stage1_selected_group_loss"] <= result.report[
        "stage1_default_group_loss"
    ] + 1e-6


def test_two_stage_runs_without_modifying_original_gptq_inner_loop():
    torch.manual_seed(3)
    weight = torch.randn(3, 64)
    state = quantize_gptq_two_stage(
        weight,
        CurvatureStats(spd(64), 20, 64),
        HIFX4Format(),
        TwoStageHiF4GPTQConfig(
            original_gptq=OriginalGPTQConfig(block_size=32),
            stage1=HiF4Stage1Config(enabled=True, scale_neighbors=2),
            stage2=HiF4Stage2Config(enabled=True, num_sweeps=1, projection_neighbors=1),
        ),
    )
    assert state.algorithm == "gptq_two_stage"
    assert state.metadata["experimental"] is True
    assert state.metadata["metadata_mode"] == "precomputed_stage1"
    assert state.format_state is not None
    assert state.report["stage2_after_loss"] <= state.report["stage2_before_loss"] + 1e-5
