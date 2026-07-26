from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

SCHEMA_VERSION = 2

SUPPORTED_CURVATURE_DOMAINS = frozenset({
    "original",
    "smoothed",
    "smooth_bf16_noqdq",
})


@dataclass(frozen=True)
class GroupLayout:
    """量化格式沿权重输入维度的逻辑分组。"""

    group_size: int = 64
    logical_shape: tuple[int, ...] = (8, 2, 4)
    axis: int = -1
    scale_levels: int = 3

    def validate(self) -> None:
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")
        product = 1
        for size in self.logical_shape:
            if size <= 0:
                raise ValueError("logical_shape entries must be positive")
            product *= int(size)
        if product != self.group_size:
            raise ValueError(
                f"logical_shape product={product}, expected group_size={self.group_size}"
            )
        if self.axis != -1:
            raise ValueError("current PTQ core only supports the last weight dimension")


@dataclass(frozen=True)
class ActivationStats:
    absmax: torch.Tensor
    num_calls: int
    channels_dim: int = -1
    module_name: str | None = None

    def validate(self, in_features: int | None = None) -> None:
        if self.absmax.ndim != 1:
            raise ValueError(f"activation absmax must be 1-D, got {tuple(self.absmax.shape)}")
        if in_features is not None and self.absmax.numel() != int(in_features):
            raise ValueError(
                f"activation stats size mismatch: got {self.absmax.numel()}, expected {in_features}"
            )
        if self.num_calls <= 0:
            raise ValueError("activation stats must contain at least one observed call")
        if not torch.isfinite(self.absmax).all():
            raise ValueError("activation absmax contains NaN or Inf")

    def to_record(self) -> dict[str, Any]:
        return {
            "absmax": self.absmax.detach().cpu().contiguous(),
            "num_calls": int(self.num_calls),
            "channels_dim": int(self.channels_dim),
            "module_name": self.module_name,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "ActivationStats":
        return cls(
            absmax=record["absmax"].detach().cpu().contiguous(),
            num_calls=int(record["num_calls"]),
            channels_dim=int(record.get("channels_dim", -1)),
            module_name=record.get("module_name"),
        )


@dataclass(frozen=True)
class CurvatureStats:
    """可由 GPTQ、Smooth 与后续算法复用的输入二阶统计。"""

    matrix: torch.Tensor
    nsamples: int
    in_features: int
    normalization: str = "mean_xtx"
    domain: str = "original"
    module_name: str | None = None

    def validate(self) -> None:
        expected = (self.in_features, self.in_features)
        if self.matrix.ndim != 2 or tuple(self.matrix.shape) != expected:
            raise ValueError(f"curvature shape={tuple(self.matrix.shape)}, expected={expected}")
        if self.nsamples <= 0:
            raise ValueError("curvature nsamples must be positive")
        if self.normalization not in {"mean_xtx", "sum_xtx"}:
            raise ValueError(f"unsupported normalization={self.normalization!r}")
        if self.domain not in SUPPORTED_CURVATURE_DOMAINS:
            raise ValueError(
                f"unsupported curvature domain={self.domain!r}; "
                f"supported={sorted(SUPPORTED_CURVATURE_DOMAINS)}"
            )
        if not torch.isfinite(self.matrix).all():
            raise ValueError("curvature contains NaN or Inf")

    def as_sum(self) -> torch.Tensor:
        self.validate()
        matrix = self.matrix.detach().float()
        return matrix * float(self.nsamples) if self.normalization == "mean_xtx" else matrix.clone()

    def as_mean(self) -> torch.Tensor:
        self.validate()
        matrix = self.matrix.detach().float()
        return matrix / float(self.nsamples) if self.normalization == "sum_xtx" else matrix.clone()

    def normalized(self) -> "CurvatureStats":
        return CurvatureStats(
            matrix=self.as_mean().cpu().contiguous(),
            nsamples=self.nsamples,
            in_features=self.in_features,
            normalization="mean_xtx",
            domain=self.domain,
            module_name=self.module_name,
        )

    def to_record(self, *, store_sum: bool = True) -> dict[str, Any]:
        matrix = self.as_sum() if store_sum else self.as_mean()
        return {
            "matrix": matrix.cpu().contiguous(),
            "nsamples": int(self.nsamples),
            "in_features": int(self.in_features),
            "normalization": "sum_xtx" if store_sum else "mean_xtx",
            "domain": self.domain,
            "module_name": self.module_name,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "CurvatureStats":
        return cls(
            matrix=record["matrix"].detach().cpu().contiguous(),
            nsamples=int(record["nsamples"]),
            in_features=int(record["in_features"]),
            normalization=str(record.get("normalization", "sum_xtx")),
            domain=str(record.get("domain", "original")),
            module_name=record.get("module_name"),
        )


@dataclass(frozen=True)
class DeviationStats:
    """两阶段实验可选的前缀量化输入偏差相关矩阵。"""

    matrix: torch.Tensor
    nsamples: int
    in_features: int
    normalization: str = "mean"
    domain: str = "original"
    module_name: str | None = None

    def validate(self) -> None:
        expected = (self.in_features, self.in_features)
        if self.matrix.ndim != 2 or tuple(self.matrix.shape) != expected:
            raise ValueError(f"deviation shape={tuple(self.matrix.shape)}, expected={expected}")
        if self.nsamples <= 0:
            raise ValueError("deviation nsamples must be positive")
        if self.normalization not in {"mean", "sum"}:
            raise ValueError("deviation normalization must be 'mean' or 'sum'")
        if not torch.isfinite(self.matrix).all():
            raise ValueError("deviation matrix contains NaN or Inf")

    def as_mean(self) -> torch.Tensor:
        self.validate()
        matrix = self.matrix.detach().float()
        return matrix / float(self.nsamples) if self.normalization == "sum" else matrix.clone()


@dataclass(frozen=True)
class ModuleCalibrationStats:
    module_name: str
    in_features: int
    activation: ActivationStats | None = None
    curvature: CurvatureStats | None = None

    def validate(self) -> None:
        if not self.module_name:
            raise ValueError("module_name cannot be empty")
        if self.in_features <= 0:
            raise ValueError("in_features must be positive")
        if self.activation is not None:
            self.activation.validate(self.in_features)
        if self.curvature is not None:
            self.curvature.validate()
            if self.curvature.in_features != self.in_features:
                raise ValueError("curvature in_features mismatch")


@dataclass
class SmoothResult:
    weight: torch.Tensor
    scale: torch.Tensor
    alpha: float
    metadata: dict[str, Any] = field(default_factory=dict)
    report: dict[str, Any] = field(default_factory=dict)


@dataclass
class MagRResult:
    """MagR 变换结果；checkpoint 仅保存轻量指标，不持久化大权重副本。"""

    weight: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)
    report: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreparedWeightState:
    weight: torch.Tensor
    curvature: CurvatureStats | None = None
    smooth_scale: torch.Tensor | None = None
    transforms: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    report: dict[str, Any] = field(default_factory=dict)
    # 若存在有损预处理（当前为 MagR），用于量化后计算相对预处理前权重的最终误差。
    quantization_reference_weight: torch.Tensor | None = None


@dataclass
class QuantizedWeightState:
    """QDQ 浮点权重及可选的真实格式状态。"""

    dequantized_weight: torch.Tensor
    format_name: str
    algorithm: str
    smooth_scale: torch.Tensor | None = None
    transforms: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    report: dict[str, Any] = field(default_factory=dict)
    format_state: dict[str, Any] | None = None

    def validate(self) -> None:
        if self.dequantized_weight.ndim != 2:
            raise ValueError("quantized Linear weight must be 2-D")
        if self.smooth_scale is not None:
            if self.smooth_scale.ndim != 1:
                raise ValueError("smooth_scale must be 1-D")
            if self.smooth_scale.numel() != self.dequantized_weight.shape[1]:
                raise ValueError("smooth_scale size mismatch")
        if not torch.isfinite(self.dequantized_weight).all():
            raise ValueError("dequantized weight contains NaN or Inf")


@dataclass
class ModuleQuantizedState:
    module_name: str
    weight: QuantizedWeightState
    bias: torch.Tensor | None = None
    in_features: int | None = None
    out_features: int | None = None

    def validate(self) -> None:
        self.weight.validate()
        out_features, in_features = self.weight.dequantized_weight.shape
        if self.in_features is not None and int(self.in_features) != in_features:
            raise ValueError("module in_features mismatch")
        if self.out_features is not None and int(self.out_features) != out_features:
            raise ValueError("module out_features mismatch")
        if self.bias is not None and self.bias.numel() != out_features:
            raise ValueError("bias size mismatch")
