from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch.nn as nn

from .adapters.module_utils import get_module
from .algorithms.gptq_original import OriginalGPTQConfig
from .algorithms.rtn import RTNConfig
from .experiments.two_stage_hifx4 import TwoStageHiF4GPTQConfig
from .formats.hifx4 import HIFX4Format
from .pipeline import (
    prepare_weight_transforms,
    quantize_prepared_gptq,
    quantize_prepared_gptq_two_stage,
    quantize_prepared_rtn,
)
from .state import ModuleCalibrationStats, ModuleQuantizedState
from .transforms.magr import MagRConfig
from .transforms.smooth import SmoothConfig


@dataclass(frozen=True)
class RTNBuildConfig:
    smooth: SmoothConfig | None = None
    magr: MagRConfig | None = None
    rtn: RTNConfig = field(default_factory=RTNConfig)
    activation_qdq: bool = True
    force_fp32: bool = True


@dataclass(frozen=True)
class GPTQBuildConfig:
    """Default builder uses the preserved repository GPTQ."""

    smooth: SmoothConfig | None = None
    magr: MagRConfig | None = None
    gptq: OriginalGPTQConfig = field(default_factory=OriginalGPTQConfig)
    activation_qdq: bool = True
    force_fp32: bool = True


@dataclass(frozen=True)
class TwoStageGPTQBuildConfig:
    smooth: SmoothConfig | None = None
    magr: MagRConfig | None = None
    two_stage: TwoStageHiF4GPTQConfig = field(default_factory=TwoStageHiF4GPTQConfig)
    activation_qdq: bool = True
    force_fp32: bool = True


def _module_state(name: str, module: nn.Linear, quantized) -> ModuleQuantizedState:
    return ModuleQuantizedState(
        module_name=name,
        weight=quantized,
        bias=None if module.bias is None else module.bias.detach().cpu(),
        in_features=module.in_features,
        out_features=module.out_features,
    )


def build_rtn_module_states(
    model: nn.Module,
    module_names: Sequence[str],
    calibration: Mapping[str, ModuleCalibrationStats],
    *,
    config: RTNBuildConfig,
) -> tuple[dict[str, ModuleQuantizedState], dict[str, Any]]:
    backend = HIFX4Format(force_fp32=config.force_fp32)
    output: dict[str, ModuleQuantizedState] = {}
    reports: list[dict[str, Any]] = []
    for name in module_names:
        module = get_module(model, name)
        if not isinstance(module, nn.Linear):
            raise TypeError(f"target {name!r} is not nn.Linear")
        stats = calibration[name]
        stats.validate()
        prepared = prepare_weight_transforms(
            module.weight.detach(),
            activation_stats=stats.activation,
            curvature=stats.curvature,
            smooth_config=config.smooth,
            magr_config=config.magr,
        )
        quantized = quantize_prepared_rtn(prepared, format_backend=backend, config=config.rtn)
        output[name] = _module_state(name, module, quantized)
        reports.append({"name": name, "metadata": quantized.metadata, "report": quantized.report})
    return output, {
        "format": "hifx4",
        "algorithm": "rtn",
        "smooth": config.smooth is not None,
        "magr": None if config.magr is None else config.magr.mode,
        "module_count": len(output),
        "modules": reports,
    }


def build_gptq_module_states(
    model: nn.Module,
    module_names: Sequence[str],
    calibration: Mapping[str, ModuleCalibrationStats],
    *,
    config: GPTQBuildConfig,
) -> tuple[dict[str, ModuleQuantizedState], dict[str, Any]]:
    backend = HIFX4Format(force_fp32=config.force_fp32)
    output: dict[str, ModuleQuantizedState] = {}
    reports: list[dict[str, Any]] = []
    for name in module_names:
        module = get_module(model, name)
        if not isinstance(module, nn.Linear):
            raise TypeError(f"target {name!r} is not nn.Linear")
        stats = calibration[name]
        stats.validate()
        if stats.curvature is None:
            raise ValueError(f"GPTQ requires curvature stats for {name!r}")
        prepared = prepare_weight_transforms(
            module.weight.detach(),
            activation_stats=stats.activation,
            curvature=stats.curvature,
            smooth_config=config.smooth,
            magr_config=config.magr,
        )
        quantized = quantize_prepared_gptq(
            prepared, format_backend=backend, config=config.gptq
        )
        output[name] = _module_state(name, module, quantized)
        reports.append({"name": name, "metadata": quantized.metadata, "report": quantized.report})
    return output, {
        "format": "hifx4",
        "algorithm": "gptq_original",
        "smooth": config.smooth is not None,
        "magr": None if config.magr is None else config.magr.mode,
        "module_count": len(output),
        "modules": reports,
    }


def build_two_stage_gptq_module_states(
    model: nn.Module,
    module_names: Sequence[str],
    calibration: Mapping[str, ModuleCalibrationStats],
    *,
    config: TwoStageGPTQBuildConfig,
) -> tuple[dict[str, ModuleQuantizedState], dict[str, Any]]:
    backend = HIFX4Format(force_fp32=config.force_fp32)
    output: dict[str, ModuleQuantizedState] = {}
    reports: list[dict[str, Any]] = []
    for name in module_names:
        module = get_module(model, name)
        if not isinstance(module, nn.Linear):
            raise TypeError(f"target {name!r} is not nn.Linear")
        stats = calibration[name]
        stats.validate()
        if stats.curvature is None:
            raise ValueError(f"two-stage GPTQ requires curvature stats for {name!r}")
        prepared = prepare_weight_transforms(
            module.weight.detach(),
            activation_stats=stats.activation,
            curvature=stats.curvature,
            smooth_config=config.smooth,
            magr_config=config.magr,
        )
        quantized = quantize_prepared_gptq_two_stage(
            prepared, format_backend=backend, config=config.two_stage
        )
        output[name] = _module_state(name, module, quantized)
        reports.append({"name": name, "metadata": quantized.metadata, "report": quantized.report})
    return output, {
        "format": "hifx4",
        "algorithm": "gptq_two_stage",
        "experimental": True,
        "smooth": config.smooth is not None,
        "magr": None if config.magr is None else config.magr.mode,
        "module_count": len(output),
        "modules": reports,
    }
