from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ...formats.hifx4 import HIFX4Format, HIFX4TensorMetadata
from ...state import CurvatureStats, DeviationStats, QuantizedWeightState
from .config import HiF4Stage2Config


@dataclass(frozen=True)
class Stage2Result:
    state: QuantizedWeightState
    report: dict[str, Any]


def _weighted_error(error: torch.Tensor, hessian: torch.Tensor) -> float:
    return float(torch.sum((error @ hessian) * error).detach().cpu().item())


def _group_order(num_groups: int, sweep: int, mode: str) -> list[int]:
    forward = list(range(num_groups))
    if mode == "forward":
        return forward
    if mode == "reverse":
        return list(reversed(forward))
    return forward if sweep % 2 == 0 else list(reversed(forward))


@torch.no_grad()
def optimize_stage2_base_scales(
    weight: torch.Tensor,
    curvature: CurvatureStats,
    original_state: QuantizedWeightState,
    format_backend: HIFX4Format,
    config: HiF4Stage2Config,
    *,
    deviation: DeviationStats | None = None,
) -> Stage2Result:
    """Fix S1P2 codes and micro-exponents, coordinate-optimize E6M2 base scales."""

    config.validate()
    curvature.validate()
    if original_state.format_state is None:
        raise ValueError("stage2 requires hifx4 format_state from original GPTQ")
    record = original_state.format_state
    if record.get("kind") != "hifx4_quantized_tensor":
        raise ValueError("unexpected format_state kind")
    metadata = HIFX4TensorMetadata.from_record(record["metadata"])
    codes = record["codes"].detach().to(weight.device, dtype=torch.int8)
    if tuple(codes.shape) != tuple(weight.shape):
        raise ValueError("stage2 codes/weight shape mismatch")

    W = weight.detach().to(dtype=torch.float32)
    H = curvature.as_mean().to(W.device, dtype=torch.float32)
    R = None
    if config.use_deviation_correlation:
        if deviation is None:
            raise ValueError("use_deviation_correlation=True requires DeviationStats")
        deviation.validate()
        R = deviation.as_mean().to(W.device, dtype=torch.float32)

    base_scales = metadata.base_scales.to(W.device, dtype=torch.float32).clone()
    lv2 = metadata.level2_exponents.to(W.device)
    lv3 = metadata.level3_exponents.to(W.device)
    basis = format_backend.basis_from_state(metadata, codes).to(W.device)
    Q = original_state.dequantized_weight.detach().to(W.device, dtype=torch.float32).clone()
    before_loss = _weighted_error(W - Q, H)
    scale_changes = 0
    scale_updates = 0
    sweep_losses: list[float] = []

    for sweep in range(config.num_sweeps):
        for group_index in _group_order(metadata.num_groups, sweep, config.group_order):
            start = group_index * metadata.group_size
            width = int(metadata.group_widths[group_index].item())
            end = start + width
            r = basis[:, start:end]
            H_rows = H[start:end, :]
            Hgg = H[start:end, start:end]
            residual = W - Q
            numerator = torch.sum((r @ H_rows) * residual, dim=1)
            if R is not None:
                # Optional paper-inspired prefix-deviation correction:
                # w^T R[:,G] r_G, evaluated independently for each output row.
                correction = torch.sum((W @ R[:, start:end]) * r, dim=1)
                numerator = numerator - correction
            denominator = torch.sum((r @ Hgg) * r, dim=1).clamp_min(1e-12)
            current = base_scales[:, group_index]
            continuous = current + numerator / denominator
            neighbors = format_backend.e6m2_neighbors(
                continuous, config.projection_neighbors
            )
            # Always keep the current legal scale as a candidate, so Stage 2
            # cannot worsen its own no-deviation coordinate objective.
            candidates = torch.cat([current[:, None], neighbors], dim=1)
            delta = candidates - current[:, None]
            loss_delta = -2.0 * delta * numerator[:, None] + delta.square() * denominator[:, None]
            best_index = torch.argmin(loss_delta, dim=1)
            selected = candidates.gather(1, best_index[:, None]).squeeze(1)
            change = selected - current
            scale_changes += int((selected != current).sum().item())
            scale_updates += int(current.numel())
            base_scales[:, group_index] = selected
            Q[:, start:end] += change[:, None] * r
        sweep_losses.append(_weighted_error(W - Q, H))

    updated_metadata = HIFX4TensorMetadata(
        base_scales=base_scales.detach().cpu(),
        level2_exponents=lv2.detach().cpu().to(torch.int8),
        level3_exponents=lv3.detach().cpu().to(torch.int8),
        group_widths=metadata.group_widths.detach().cpu(),
        in_features=metadata.in_features,
        group_size=metadata.group_size,
    )
    updated_metadata.validate()
    after_loss = _weighted_error(W - Q, H)

    state = QuantizedWeightState(
        dequantized_weight=Q.to(
            original_state.dequantized_weight.device,
            dtype=original_state.dequantized_weight.dtype,
        ).contiguous(),
        format_name=original_state.format_name,
        algorithm="gptq_two_stage",
        smooth_scale=original_state.smooth_scale,
        transforms=list(original_state.transforms),
        metadata={
            **original_state.metadata,
            "experimental": True,
            "two_stage_stage2": True,
            "stage2_sweeps": int(config.num_sweeps),
            "stage2_use_deviation": bool(config.use_deviation_correlation),
        },
        report={
            **original_state.report,
            "stage2_before_loss": before_loss,
            "stage2_after_loss": after_loss,
            "stage2_improvement": before_loss - after_loss,
            "stage2_scale_change_rate": scale_changes / max(scale_updates, 1),
            "stage2_sweep_losses": sweep_losses,
        },
        format_state={
            "kind": "hifx4_quantized_tensor",
            "metadata": updated_metadata.to_record(),
            "codes": codes.detach().cpu().contiguous(),
        },
    )
    state.validate()
    return Stage2Result(state=state, report=dict(state.report))
