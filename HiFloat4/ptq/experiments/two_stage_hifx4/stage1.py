from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ...formats.hifx4 import HIFX4Format, HIFX4GroupState, HIFX4TensorMetadata
from ...state import CurvatureStats
from .config import HiF4Stage1Config


@dataclass(frozen=True)
class Stage1Result:
    metadata: HIFX4TensorMetadata
    report: dict[str, Any]


def _row_weighted_loss(error: torch.Tensor, hessian: torch.Tensor) -> torch.Tensor:
    return torch.sum((error @ hessian) * error, dim=1)


@torch.no_grad()
def optimize_stage1_metadata(
    weight: torch.Tensor,
    curvature: CurvatureStats,
    format_backend: HIFX4Format,
    config: HiF4Stage1Config,
) -> Stage1Result:
    """Hessian-aware legal E6M2 neighbourhood search before GPTQ.

    For each output row and 64-column group, the base scale is searched in a
    legal E6M2 neighbourhood. Micro-exponents and S1P2 codes are recomputed for
    every candidate. The GPTQ inner loop later freezes the selected metadata.
    """

    config.validate()
    curvature.validate()
    if weight.ndim != 2:
        raise ValueError("stage1 weight must be 2-D")
    if curvature.in_features != weight.shape[1]:
        raise ValueError("stage1 curvature/weight mismatch")

    W = weight.detach().float()
    H = curvature.as_mean().to(W.device, dtype=torch.float32)
    group_size = format_backend.group_layout.group_size
    states: list[HIFX4GroupState] = []
    default_total = 0.0
    selected_total = 0.0
    changed_rows = 0
    total_rows = 0
    candidate_count = 2 * config.scale_neighbors + 1

    for group_start in range(0, W.shape[1], group_size):
        group_end = min(group_start + group_size, W.shape[1])
        group = W[:, group_start:group_end]
        Hgg = H[group_start:group_end, group_start:group_end]
        default = format_backend.encode_group(group, group_start=group_start)
        default_error = group - default.dequantized.float()
        default_loss = _row_weighted_loss(default_error, Hgg)

        candidates = format_backend.e6m2_neighbors(
            default.metadata.base_scale, config.scale_neighbors
        )
        best_loss = default_loss.clone()
        best_base = default.metadata.base_scale.clone()
        best_lv2 = default.metadata.level2_exponents.clone()
        best_lv3 = default.metadata.level3_exponents.clone()

        for candidate_index in range(candidates.shape[1]):
            encoded = format_backend.encode_group(
                group,
                base_scale=candidates[:, candidate_index],
                group_start=group_start,
            )
            loss = _row_weighted_loss(group - encoded.dequantized.float(), Hgg)
            better = loss < best_loss
            if torch.any(better):
                best_loss = torch.where(better, loss, best_loss)
                best_base = torch.where(better, encoded.metadata.base_scale, best_base)
                best_lv2 = torch.where(
                    better[:, None], encoded.metadata.level2_exponents, best_lv2
                )
                best_lv3 = torch.where(
                    better[:, None], encoded.metadata.level3_exponents, best_lv3
                )

        full_scale = format_backend.full_scale_from_metadata(
            best_base, best_lv2, best_lv3, group.shape[1]
        )
        states.append(
            HIFX4GroupState(
                base_scale=best_base,
                level2_exponents=best_lv2,
                level3_exponents=best_lv3,
                full_scale=full_scale,
                width=group.shape[1],
                group_start=group_start,
            )
        )
        default_total += float(default_loss.sum().item())
        selected_total += float(best_loss.sum().item())
        changed_rows += int((best_base != default.metadata.base_scale).sum().item())
        total_rows += int(best_base.numel())

    metadata = HIFX4TensorMetadata.from_group_states(states, in_features=W.shape[1])
    return Stage1Result(
        metadata=metadata,
        report={
            "stage1_candidate_count": candidate_count,
            "stage1_default_group_loss": default_total,
            "stage1_selected_group_loss": selected_total,
            "stage1_improvement": default_total - selected_total,
            "stage1_scale_change_rate": changed_rows / max(total_rows, 1),
        },
    )
