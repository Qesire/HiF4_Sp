from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..formats.base import WeightFormatBackend
from ..formats.hifx4 import HIFX4Format, HIFX4GroupState, HIFX4TensorMetadata
from ..state import CurvatureStats, QuantizedWeightState


@dataclass(frozen=True)
class OriginalGPTQConfig:
    """原仓库 static-metadata columnwise GPTQ 配置。"""

    block_size: int = 64
    damp_percent: float = 0.01
    act_order: bool = False
    max_cholesky_retries: int = 5
    damp_growth: float = 10.0
    min_damp: float = 1e-8
    symmetrize_hessian: bool = True
    compute_device: str | None = None

    def validate(self) -> None:
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.damp_percent < 0:
            raise ValueError("damp_percent cannot be negative")
        if self.act_order:
            raise ValueError(
                "global act-order is not included in the preserved hifx4 reference path"
            )
        if self.max_cholesky_retries < 0:
            raise ValueError("max_cholesky_retries cannot be negative")
        if self.damp_growth <= 1.0:
            raise ValueError("damp_growth must be greater than 1")
        if self.min_damp < 0:
            raise ValueError("min_damp cannot be negative")
        if self.compute_device is not None and not str(self.compute_device).strip():
            raise ValueError("compute_device cannot be empty")


# Backward-compatible name used by the first scaffold.
GPTQConfig = OriginalGPTQConfig


def _weighted_error(error: torch.Tensor, hessian: torch.Tensor) -> float:
    return float(torch.sum((error @ hessian) * error).detach().cpu().item())


def _build_inverse_cholesky(
    hessian: torch.Tensor,
    *,
    initial_damp: float,
    config: OriginalGPTQConfig,
) -> tuple[torch.Tensor, float, int]:
    identity = torch.eye(hessian.shape[0], device=hessian.device, dtype=hessian.dtype)
    damp = max(float(initial_damp), float(config.min_damp))
    last_error: RuntimeError | None = None
    for attempt in range(config.max_cholesky_retries + 1):
        try:
            chol = torch.linalg.cholesky(hessian + identity * damp)
            inverse = torch.cholesky_inverse(chol)
            return torch.linalg.cholesky(inverse, upper=True), damp, attempt
        except RuntimeError as exc:
            last_error = exc
            damp = max(damp * config.damp_growth, config.min_damp)
    raise RuntimeError(
        "GPTQ Hessian Cholesky failed after damping retries; "
        f"last_damp={damp}, shape={tuple(hessian.shape)}"
    ) from last_error


@torch.no_grad()
def quantize_gptq_original(
    weight: torch.Tensor,
    curvature: CurvatureStats,
    format_backend: WeightFormatBackend,
    config: OriginalGPTQConfig | None = None,
    *,
    precomputed_metadata: HIFX4TensorMetadata | None = None,
) -> QuantizedWeightState:
    """Preserve the repository's frozen-metadata, columnwise GPTQ behaviour.

    Without ``precomputed_metadata`` each format group is prepared from the global
    working weight ``W`` at the moment the group boundary is reached. Crucially,
    it is *not* recomputed from the block-local ``W1`` after preceding columns in
    the same lazy block have been corrected. This matches the repository path.

    ``precomputed_metadata`` is used only by the experimental Stage-1 wrapper;
    the GPTQ inner loop itself remains unchanged.
    """

    if weight.ndim != 2 or not torch.is_floating_point(weight):
        raise ValueError("GPTQ weight must be a floating-point 2-D tensor")
    curvature.validate()
    out_features, in_features = map(int, weight.shape)
    if curvature.in_features != in_features:
        raise ValueError("curvature/weight in_features mismatch")

    cfg = config or OriginalGPTQConfig()
    cfg.validate()
    group_size = int(format_backend.group_layout.group_size)
    if precomputed_metadata is not None:
        precomputed_metadata.validate()
        if precomputed_metadata.in_features != in_features:
            raise ValueError("precomputed metadata in_features mismatch")
        if precomputed_metadata.group_size != group_size:
            raise ValueError("precomputed metadata group_size mismatch")
        if not isinstance(format_backend, HIFX4Format):
            raise TypeError("precomputed hifx4 metadata requires HIFX4Format")

    original_device = weight.device
    original_dtype = weight.dtype
    work_device = original_device if cfg.compute_device is None else torch.device(cfg.compute_device)

    W = weight.detach().to(work_device, dtype=torch.float32).clone()
    H_original = curvature.as_mean().to(work_device, dtype=torch.float32).clone()
    H = H_original.clone()
    if cfg.symmetrize_hessian:
        H = 0.5 * (H + H.T)
        H_original = 0.5 * (H_original + H_original.T)

    dead = torch.diag(H) == 0
    dead_count = int(dead.sum().item())
    if dead_count:
        indices = torch.nonzero(dead, as_tuple=False).flatten()
        H[indices, indices] = 1.0
        W[:, indices] = 0.0

    requested_damp = cfg.damp_percent * float(torch.diag(H).mean().item())
    Hinv, applied_damp, retry_count = _build_inverse_cholesky(
        H, initial_damp=requested_damp, config=cfg
    )

    Q = torch.zeros_like(W)
    codes = torch.zeros_like(W, dtype=torch.int8)
    block_count = 0
    group_states: list[HIFX4GroupState] = []
    active_group_state: Any | None = None
    active_group_start = -1
    supports_codes = hasattr(format_backend, "quantize_column_with_code")

    for block_start in range(0, in_features, cfg.block_size):
        block_end = min(block_start + cfg.block_size, in_features)
        block_count += 1
        W1 = W[:, block_start:block_end].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[block_start:block_end, block_start:block_end]

        for local_col in range(block_end - block_start):
            global_col = block_start + local_col
            d = Hinv1[local_col, local_col]
            if not torch.isfinite(d) or float(torch.abs(d).item()) == 0.0:
                raise RuntimeError(f"invalid inverse-Hessian diagonal at column {global_col}")

            group_start = (global_col // group_size) * group_size
            if group_start != active_group_start:
                group_end = min(group_start + group_size, in_features)
                group_index = group_start // group_size
                if precomputed_metadata is not None:
                    active_group_state = precomputed_metadata.group_state(
                        group_index,
                        format_backend=format_backend,
                        device=work_device,
                    )
                else:
                    # Reference behaviour: use global W, not the already corrected W1.
                    active_group_state = format_backend.prepare_group(
                        W[:, group_start:group_end].clone(),
                        group_start=group_start,
                    )
                active_group_start = group_start
                if isinstance(active_group_state, HIFX4GroupState):
                    group_states.append(active_group_state)

            if active_group_state is None:
                raise RuntimeError("format group state was not initialized")
            w = W1[:, local_col]
            local_group_col = global_col - active_group_start
            if supports_codes:
                q, code = format_backend.quantize_column_with_code(
                    w, active_group_state, local_group_col
                )
            else:
                q = format_backend.quantize_column(w, active_group_state, local_group_col)
                code = None
            q = q.to(W1.device, dtype=W1.dtype)
            if q.shape != w.shape or not torch.isfinite(q).all():
                raise ValueError(f"invalid quantized column {global_col}")
            Q1[:, local_col] = q
            if code is not None:
                codes[:, global_col] = code.to(codes.device, dtype=torch.int8)

            error = (w - q) / d
            W1[:, local_col:] -= error.unsqueeze(1) @ Hinv1[
                local_col, local_col:
            ].unsqueeze(0)
            Err1[:, local_col] = error

        Q[:, block_start:block_end] = Q1
        if block_end < in_features:
            W[:, block_end:] -= Err1 @ Hinv[block_start:block_end, block_end:]

    effective_weight = weight.detach().to(work_device, dtype=torch.float32).clone()
    if dead_count:
        effective_weight[:, dead] = 0.0
    error = effective_weight - Q
    relative_error = float(
        (torch.linalg.vector_norm(error) / torch.linalg.vector_norm(effective_weight).clamp_min(1e-12))
        .detach().cpu().item()
    )

    format_state: dict[str, Any] | None = None
    if isinstance(format_backend, HIFX4Format) and len(group_states) == math_ceil_div(in_features, group_size):
        tensor_metadata = HIFX4TensorMetadata.from_group_states(
            group_states, in_features=in_features
        )
        format_state = {
            "kind": "hifx4_quantized_tensor",
            "metadata": tensor_metadata.to_record(),
            "codes": codes.detach().cpu().contiguous(),
        }

    state = QuantizedWeightState(
        dequantized_weight=Q.to(original_device, dtype=original_dtype).contiguous(),
        format_name=format_backend.name,
        algorithm="gptq_original",
        metadata={
            "gptq_variant": "repository_static_metadata_columnwise",
            "metadata_mode": "precomputed_stage1" if precomputed_metadata is not None else "repository_reference",
            "gptq_block_size": int(cfg.block_size),
            "format_group_size": group_size,
            "damp_percent": float(cfg.damp_percent),
            "applied_damp": float(applied_damp),
            "cholesky_retries": int(retry_count),
            "dead_columns": dead_count,
            "block_count": block_count,
            "curvature_domain": curvature.domain,
            **format_backend.metadata(),
        },
        report={
            "relative_weight_error": relative_error,
            "hessian_weighted_error": _weighted_error(error, H_original),
        },
        format_state=format_state,
    )
    state.validate()
    return state


def math_ceil_div(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


# Compatibility with the first scaffold and external imports.
quantize_gptq = quantize_gptq_original

__all__ = [
    "GPTQConfig",
    "OriginalGPTQConfig",
    "quantize_gptq",
    "quantize_gptq_original",
]
