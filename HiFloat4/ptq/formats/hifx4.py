from __future__ import annotations

import importlib
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from .base import WeightFormatBackend
from ..state import GroupLayout


@dataclass(frozen=True)
class HIFX4GroupState:
    """一个连续 hifx4 group 的冻结 metadata。"""

    base_scale: torch.Tensor                 # [out_features]
    level2_exponents: torch.Tensor           # [out_features, 8], int8
    level3_exponents: torch.Tensor           # [out_features, 16], int8
    full_scale: torch.Tensor                 # [out_features, width]
    width: int
    group_start: int
    man_bits: int = 3

    def validate(self) -> None:
        if self.base_scale.ndim != 1:
            raise ValueError("base_scale must be 1-D")
        out_features = self.base_scale.shape[0]
        if tuple(self.level2_exponents.shape) != (out_features, 8):
            raise ValueError("level2_exponents must have shape [out_features, 8]")
        if tuple(self.level3_exponents.shape) != (out_features, 16):
            raise ValueError("level3_exponents must have shape [out_features, 16]")
        if tuple(self.full_scale.shape) != (out_features, self.width):
            raise ValueError("full_scale shape mismatch")
        if self.width <= 0 or self.width > 64:
            raise ValueError("hifx4 group width must be in [1, 64]")
        if self.man_bits != 3:
            raise ValueError("hifx4 S1P2 path expects man_bits=3")

    def to(self, device: str | torch.device) -> "HIFX4GroupState":
        target = torch.device(device)
        return HIFX4GroupState(
            base_scale=self.base_scale.to(target),
            level2_exponents=self.level2_exponents.to(target),
            level3_exponents=self.level3_exponents.to(target),
            full_scale=self.full_scale.to(target),
            width=self.width,
            group_start=self.group_start,
            man_bits=self.man_bits,
        )


@dataclass(frozen=True)
class HIFX4EncodedGroup:
    dequantized: torch.Tensor
    codes: torch.Tensor                      # signed int8, decoded value = codes / 4
    metadata: HIFX4GroupState

    def validate(self) -> None:
        self.metadata.validate()
        if self.dequantized.shape != self.codes.shape:
            raise ValueError("codes/dequantized shape mismatch")
        if self.dequantized.shape != self.metadata.full_scale.shape:
            raise ValueError("encoded group shape mismatch")
        if self.codes.dtype != torch.int8:
            raise TypeError("hifx4 codes must be int8")


@dataclass(frozen=True)
class HIFX4TensorMetadata:
    """按原始列顺序保存全部 64-group metadata。"""

    base_scales: torch.Tensor                # [out_features, num_groups]
    level2_exponents: torch.Tensor           # [out_features, num_groups, 8]
    level3_exponents: torch.Tensor           # [out_features, num_groups, 16]
    group_widths: torch.Tensor               # [num_groups]
    in_features: int
    group_size: int = 64

    def validate(self) -> None:
        if self.base_scales.ndim != 2:
            raise ValueError("base_scales must be 2-D")
        out_features, num_groups = self.base_scales.shape
        if tuple(self.level2_exponents.shape) != (out_features, num_groups, 8):
            raise ValueError("level2 metadata shape mismatch")
        if tuple(self.level3_exponents.shape) != (out_features, num_groups, 16):
            raise ValueError("level3 metadata shape mismatch")
        if tuple(self.group_widths.shape) != (num_groups,):
            raise ValueError("group_widths shape mismatch")
        if int(self.group_widths.sum().item()) != self.in_features:
            raise ValueError("group widths do not cover in_features")
        if torch.any(self.group_widths <= 0) or torch.any(self.group_widths > self.group_size):
            raise ValueError("invalid group width")

    @property
    def num_groups(self) -> int:
        return int(self.base_scales.shape[1])

    def group_state(
        self,
        group_index: int,
        *,
        format_backend: "HIFX4Format",
        device: str | torch.device,
    ) -> HIFX4GroupState:
        self.validate()
        idx = int(group_index)
        if idx < 0 or idx >= self.num_groups:
            raise IndexError(idx)
        width = int(self.group_widths[idx].item())
        base = self.base_scales[:, idx].to(device=device, dtype=torch.float32)
        lv2 = self.level2_exponents[:, idx].to(device=device)
        lv3 = self.level3_exponents[:, idx].to(device=device)
        full = format_backend.full_scale_from_metadata(base, lv2, lv3, width)
        state = HIFX4GroupState(
            base_scale=base,
            level2_exponents=lv2.to(torch.int8),
            level3_exponents=lv3.to(torch.int8),
            full_scale=full,
            width=width,
            group_start=idx * self.group_size,
        )
        state.validate()
        return state

    def to_record(self) -> dict[str, Any]:
        self.validate()
        return {
            "base_scales": self.base_scales.detach().cpu().float().contiguous(),
            "level2_exponents": self.level2_exponents.detach().cpu().to(torch.int8).contiguous(),
            "level3_exponents": self.level3_exponents.detach().cpu().to(torch.int8).contiguous(),
            "group_widths": self.group_widths.detach().cpu().to(torch.int32).contiguous(),
            "in_features": int(self.in_features),
            "group_size": int(self.group_size),
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "HIFX4TensorMetadata":
        value = cls(
            base_scales=record["base_scales"].detach().cpu().float().contiguous(),
            level2_exponents=record["level2_exponents"].detach().cpu().to(torch.int8).contiguous(),
            level3_exponents=record["level3_exponents"].detach().cpu().to(torch.int8).contiguous(),
            group_widths=record["group_widths"].detach().cpu().to(torch.int32).contiguous(),
            in_features=int(record["in_features"]),
            group_size=int(record.get("group_size", 64)),
        )
        value.validate()
        return value

    @classmethod
    def from_group_states(
        cls,
        states: Iterable[HIFX4GroupState],
        *,
        in_features: int,
    ) -> "HIFX4TensorMetadata":
        items = list(states)
        if not items:
            raise ValueError("at least one group state is required")
        for state in items:
            state.validate()
        value = cls(
            base_scales=torch.stack([s.base_scale.detach().cpu().float() for s in items], dim=1),
            level2_exponents=torch.stack(
                [s.level2_exponents.detach().cpu().to(torch.int8) for s in items], dim=1
            ),
            level3_exponents=torch.stack(
                [s.level3_exponents.detach().cpu().to(torch.int8) for s in items], dim=1
            ),
            group_widths=torch.tensor([s.width for s in items], dtype=torch.int32),
            in_features=int(in_features),
        )
        value.validate()
        return value


_API_CACHE: tuple[Any, Any, str] | None = None


def _add_hif4_repo_to_path() -> None:
    repo = os.environ.get("HIF4_MAIN_REPO") or os.environ.get("HIF4_REPO")
    if not repo:
        return
    root = Path(repo).expanduser().resolve()
    for path in (root, root / "HiFloat4", root / "HiFloat4" / "hif4_gpu"):
        text = str(path)
        if path.exists() and text not in sys.path:
            sys.path.insert(0, text)


def load_hif4_api() -> tuple[Any, Any, str]:
    global _API_CACHE
    if _API_CACHE is not None:
        return _API_CACHE
    _add_hif4_repo_to_path()
    candidates = ["HiFloat4.hif4_gpu.quant_cy", "hif4_gpu.quant_cy", "HiFloat4.main"]
    errors: list[tuple[str, str]] = []
    for name in candidates:
        try:
            module = importlib.import_module(name)
            if hasattr(module, "QType") and hasattr(module, "quant_dequant_float"):
                _API_CACHE = (module.QType, module.quant_dequant_float, name)
                return _API_CACHE
        except Exception as exc:  # pragma: no cover
            errors.append((name, repr(exc)))
    raise ImportError("Cannot import HiF4 QType/quant_dequant_float: " + repr(errors))


class HIFX4Format(WeightFormatBackend):
    """hifx4 格式后端。

    官方 CUDA API 只用于完整 tensor QDQ；metadata、逐列条件量化和两阶段实验
    使用同一套 8×2×4 层级数学，可在 CPU 上执行。
    """

    name = "hifx4"
    group_layout = GroupLayout(group_size=64, logical_shape=(8, 2, 4), axis=-1, scale_levels=3)
    man_bits = 3
    min_base_scale = 2 ** -48
    max_base_scale = 49152.0

    def __init__(self, *, force_fp32: bool = True) -> None:
        self.force_fp32 = bool(force_fp32)
        self.group_layout.validate()

    @staticmethod
    def _bf16_round(x: torch.Tensor) -> torch.Tensor:
        return x.to(torch.bfloat16).to(torch.float32)

    @classmethod
    def e6m2_round(cls, x: torch.Tensor) -> torch.Tensor:
        work = x.float().clamp(min=cls.min_base_scale, max=cls.max_base_scale)
        exponent = torch.floor(torch.log2(work))
        rounded = torch.round(work * torch.exp2(2 - exponent)) * torch.exp2(exponent - 2)
        return rounded.clamp(min=cls.min_base_scale, max=cls.max_base_scale)

    @classmethod
    def default_base_scale(cls, weight_group: torch.Tensor) -> torch.Tensor:
        if weight_group.ndim != 2:
            raise ValueError("weight_group must be 2-D")
        width = int(weight_group.shape[1])
        if width <= 0 or width > 64:
            raise ValueError("hifx4 group width must be in [1, 64]")
        work = weight_group.float()
        if width < 64:
            work = F.pad(work, (0, 64 - width), value=0.0)
        grouped = work.reshape(work.shape[0], 8, 2, 4)
        max_lv1 = grouped.abs().amax(dim=(-1, -2, -3), keepdim=False)
        div7 = cls._bf16_round(torch.ones_like(max_lv1) / 7.0)
        scale = cls._bf16_round(max_lv1 * div7).clamp(
            min=cls.min_base_scale, max=cls.max_base_scale
        )
        exponent = torch.floor(torch.log2(scale))
        mantissa = scale / torch.exp2(exponent) * 2 ** 7
        scale = torch.round(mantissa) / 2 ** 7 * torch.exp2(exponent)
        return cls.e6m2_round(scale)

    @classmethod
    def _micro_exponents(
        cls,
        weight_group: torch.Tensor,
        base_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        width = int(weight_group.shape[1])
        work = weight_group.float()
        if width < 64:
            work = F.pad(work, (0, 64 - width), value=0.0)
        grouped = work.reshape(work.shape[0], 8, 2, 4)
        unsigned = grouped.abs()
        max_lv3 = unsigned.amax(dim=-1, keepdim=True)
        max_lv2 = max_lv3.amax(dim=-2, keepdim=True)

        base = base_scale.float().reshape(-1, 1, 1, 1).clamp_min(cls.min_base_scale)
        reciprocal = cls._bf16_round(base.reciprocal())
        lv2 = torch.floor((max_lv2 * reciprocal).clamp(0, 4) / 4).to(torch.int8)
        lv2_scale = torch.exp2(lv2.float())
        lv3 = torch.floor((max_lv3 * reciprocal / lv2_scale).clamp(0, 2) / 2).to(torch.int8)
        return lv2.reshape(-1, 8), lv3.reshape(-1, 16)

    @staticmethod
    def full_scale_from_metadata(
        base_scale: torch.Tensor,
        level2_exponents: torch.Tensor,
        level3_exponents: torch.Tensor,
        width: int,
    ) -> torch.Tensor:
        out_features = int(base_scale.numel())
        if tuple(level2_exponents.shape) != (out_features, 8):
            raise ValueError("level2 exponent shape mismatch")
        if tuple(level3_exponents.shape) != (out_features, 16):
            raise ValueError("level3 exponent shape mismatch")
        lv2 = level2_exponents.float().reshape(out_features, 8, 1, 1).expand(-1, -1, 2, 4)
        lv3 = level3_exponents.float().reshape(out_features, 8, 2, 1).expand(-1, -1, -1, 4)
        scale = base_scale.float().reshape(out_features, 1, 1, 1) * torch.exp2(lv2 + lv3)
        return scale.reshape(out_features, 64)[:, :width].contiguous()

    @classmethod
    def encode_group(
        cls,
        weight_group: torch.Tensor,
        *,
        base_scale: torch.Tensor | None = None,
        group_start: int = 0,
    ) -> HIFX4EncodedGroup:
        if weight_group.ndim != 2:
            raise ValueError("weight_group must be 2-D")
        width = int(weight_group.shape[1])
        if width <= 0 or width > 64:
            raise ValueError("hifx4 group width must be in [1, 64]")
        base = cls.default_base_scale(weight_group) if base_scale is None else cls.e6m2_round(base_scale)
        base = base.to(device=weight_group.device, dtype=torch.float32).reshape(-1)
        if base.numel() != weight_group.shape[0]:
            raise ValueError("base_scale must have one value per output row")
        lv2, lv3 = cls._micro_exponents(weight_group, base)
        full_scale = cls.full_scale_from_metadata(base, lv2, lv3, width).to(weight_group.device)
        normalized = weight_group.float() / full_scale.clamp_min(cls.min_base_scale)
        codes = torch.round(normalized * 4.0).clamp(-7, 7).to(torch.int8)
        dequantized = (codes.float() / 4.0 * full_scale).to(weight_group.dtype)
        metadata = HIFX4GroupState(
            base_scale=base,
            level2_exponents=lv2.to(weight_group.device),
            level3_exponents=lv3.to(weight_group.device),
            full_scale=full_scale,
            width=width,
            group_start=int(group_start),
            man_bits=cls.man_bits,
        )
        result = HIFX4EncodedGroup(dequantized=dequantized, codes=codes, metadata=metadata)
        result.validate()
        return result

    @classmethod
    def e6m2_neighbors(cls, center: torch.Tensor, radius: int) -> torch.Tensor:
        """Return sorted legal E6M2 neighbours per row, shape [rows, candidates]."""
        if radius < 0:
            raise ValueError("radius cannot be negative")
        values = cls.e6m2_round(center.detach().float().reshape(-1))
        candidate_count = 2 * radius + 1
        output = torch.empty((values.numel(), candidate_count), device=values.device)
        mantissas = torch.tensor([1.0, 1.25, 1.5, 1.75], device=values.device)
        for row, value in enumerate(values):
            exponent = int(torch.floor(torch.log2(value)).item())
            pool: list[float] = []
            extent = max(3, math.ceil((radius + 2) / 4) + 2)
            for exp in range(exponent - extent, exponent + extent + 1):
                for mantissa in mantissas.tolist():
                    candidate = mantissa * (2.0 ** exp)
                    if cls.min_base_scale <= candidate <= cls.max_base_scale:
                        pool.append(candidate)
            pool = sorted(set(pool))
            nearest = min(range(len(pool)), key=lambda i: abs(pool[i] - float(value.item())))
            lo = max(0, nearest - radius)
            hi = min(len(pool), nearest + radius + 1)
            selected = pool[lo:hi]
            while len(selected) < candidate_count:
                if lo > 0:
                    lo -= 1
                    selected.insert(0, pool[lo])
                elif hi < len(pool):
                    selected.append(pool[hi])
                    hi += 1
                else:
                    selected.append(selected[-1])
            output[row] = torch.tensor(selected[:candidate_count], device=values.device)
        return output

    def quantize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if not tensor.is_cuda:
            raise RuntimeError(f"hifx4 QDQ requires CUDA tensor, got {tensor.device}")
        QType, quant_dequant_float, _ = load_hif4_api()
        qparams = QType("hifx4").dim(-1)
        return quant_dequant_float(
            tensor.contiguous(), qparams, force_fp32=self.force_fp32
        ).to(tensor.dtype)

    @torch.no_grad()
    def quantize_weight(
        self,
        weight: torch.Tensor,
        *,
        quant_device: str | torch.device = "cuda",
        return_device: str | torch.device | None = None,
    ) -> torch.Tensor:
        if not torch.cuda.is_available():
            raise RuntimeError("hifx4 weight QDQ requires CUDA")
        target = weight.device if return_device is None else torch.device(return_device)
        working = weight.detach().to(device=quant_device).contiguous()
        result = self.quantize_tensor(working).to(target, dtype=weight.dtype).contiguous()
        del working
        return result

    def quantize_activation(self, x: torch.Tensor) -> torch.Tensor:
        return self.quantize_tensor(x)

    def prepare_group(
        self,
        weight_group: torch.Tensor,
        *,
        group_start: int = 0,
    ) -> HIFX4GroupState:
        return self.encode_group(weight_group, group_start=group_start).metadata

    def quantize_column_with_code(
        self,
        column: torch.Tensor,
        group_state: HIFX4GroupState,
        local_column_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        group_state.validate()
        index = int(local_column_index)
        if index < 0 or index >= group_state.width:
            raise IndexError(index)
        original_shape = column.shape
        values = column[:, None] if column.ndim == 1 else column
        if values.ndim != 2 or values.shape[1] != 1:
            raise ValueError("column must have shape [out_features] or [out_features, 1]")
        scale = group_state.full_scale[:, index : index + 1].to(values.device).float()
        codes = torch.round(values.float() / scale.clamp_min(self.min_base_scale) * 4.0)
        codes = codes.clamp(-7, 7).to(torch.int8)
        quantized = (codes.float() / 4.0 * scale).to(column.dtype).reshape(original_shape)
        return quantized, codes.reshape(original_shape)

    def quantize_column(
        self,
        column: torch.Tensor,
        group_state: HIFX4GroupState,
        local_column_index: int,
    ) -> torch.Tensor:
        return self.quantize_column_with_code(column, group_state, local_column_index)[0]

    @staticmethod
    def basis_from_state(
        metadata: HIFX4TensorMetadata,
        codes: torch.Tensor,
    ) -> torch.Tensor:
        metadata.validate()
        if tuple(codes.shape) != (metadata.base_scales.shape[0], metadata.in_features):
            raise ValueError("codes shape mismatch")
        result = torch.empty_like(codes, dtype=torch.float32)
        for group_index in range(metadata.num_groups):
            start = group_index * metadata.group_size
            width = int(metadata.group_widths[group_index].item())
            end = start + width
            lv2 = metadata.level2_exponents[:, group_index].float().reshape(-1, 8, 1, 1)
            lv3 = metadata.level3_exponents[:, group_index].float().reshape(-1, 8, 2, 1)
            hierarchy = torch.exp2(lv2 + lv3).expand(-1, -1, -1, 4).reshape(codes.shape[0], 64)[:, :width]
            result[:, start:end] = codes[:, start:end].float() / 4.0 * hierarchy
        return result

    def metadata(self) -> dict[str, Any]:
        api_name = "lazy"
        if _API_CACHE is not None:
            api_name = _API_CACHE[2]
        return {
            "format": self.name,
            "qtype": "hifx4",
            "group_size": 64,
            "logical_shape": [8, 2, 4],
            "scale_levels": 3,
            "element_code": "S1P2_signed_quarter_steps",
            "force_fp32": self.force_fp32,
            "storage": "float_qdq",
            "api_module": api_name,
        }
