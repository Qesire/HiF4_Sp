from __future__ import annotations

"""Wan2.2-I2V dual-expert adapter for the generic hifx4 PTQ package.

The adapter deliberately reuses the *existing Wan native HiF4 backend* for
weight and activation QDQ.  This package only supplies the calibration
conversion and the optional SmoothQuant / MagR transforms.

Runtime order for a transformed Linear is:

    x -> x / smooth_scale (optional) -> native HiF4 activation QDQ
      -> F.linear(native-HiF4-QDQ(preprocessed_weight), bias)

That order is required for SmoothQuant floating-point equivalence.
"""

import argparse
import atexit
import importlib
import inspect
import json
import logging
import re
from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..adapters.module_utils import get_module, set_module
from ..pipeline import prepare_weight_transforms
from ..algorithms.gptq_original import OriginalGPTQConfig, quantize_gptq_original
from ..formats.hifx4 import HIFX4Format
from ..state import (
    ActivationStats,
    CurvatureStats,
    MagRResult,
    PreparedWeightState,
)
from ..transforms.hessian_domain import transform_curvature_for_smooth
from ..transforms.magr import MagRConfig, apply_magr
from ..transforms.smooth import SmoothConfig

LOGGER = logging.getLogger(__name__)

WAN_EXPERTS = ("low_noise_model", "high_noise_model")
_PREPROCESS_CHOICES = ("none", "smooth", "magr", "smooth_magr")
_WEIGHT_QUANTIZERS = ("rtn", "gptq")
_MISSING_POLICIES = ("error", "hif4_rtn", "bf16")
_RUNTIME_COUNTERS: dict[str, int] = {
    "weight_qdq_calls": 0,
    "activation_qdq_calls": 0,
    "linear_forward_calls": 0,
}
_RUNTIME_REPORT_CONTEXT: dict[str, Any] = {}


def reset_wan_hif4_runtime_counters() -> None:
    for key in _RUNTIME_COUNTERS:
        _RUNTIME_COUNTERS[key] = 0


def get_wan_hif4_runtime_counters() -> dict[str, int]:
    return dict(_RUNTIME_COUNTERS)


def write_wan_hif4_runtime_report(path: str | Path, **extra: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "__meta__": {
            "adapter": "HiFloat4.ptq.adapters.wan_i2v",
            "format": "hifx4",
            **_RUNTIME_REPORT_CONTEXT,
            **extra,
        },
        **get_wan_hif4_runtime_counters(),
    }
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

# The successful Wan HiF4 baseline selected the DiT block Linear modules and
# skipped non-residual/control paths.  On Wan2.2-I2V-A14B this resolves to the
# established 400 modules per expert.
_DEFAULT_SKIP_TOKENS = (
    "norm",
    "time",
    "embed",
    "embedding",
    "modulation",
    "modulate",
    "adaln",
    "head",
    "final",
    "vae",
    "t5",
    "text",
)


@dataclass(frozen=True)
class WanHiF4PreprocessConfig:
    preprocess: str = "none"
    smooth: SmoothConfig = field(default_factory=SmoothConfig)
    magr: MagRConfig = field(
        default_factory=lambda: MagRConfig(
            mode="group64",
            alpha=1e-4,
            outer_iters=200,
            spectral_method="exact",
            power_iters=32,
            compute_device="cuda",
        )
    )
    weight_quantizer: str = "rtn"
    gptq: OriginalGPTQConfig = field(default_factory=OriginalGPTQConfig)
    activation_qdq: bool = True
    force_fp32: bool = True
    include_regex: str = ""
    exclude_regex: str = ""
    quant_device: str = "cuda"
    strict_artifacts: bool = True
    missing_policy: str = "error"
    experts: tuple[str, ...] = WAN_EXPERTS

    def validate(self) -> None:
        if self.preprocess not in _PREPROCESS_CHOICES:
            raise ValueError(
                f"preprocess must be one of {_PREPROCESS_CHOICES}, got {self.preprocess!r}"
            )
        self.smooth.validate()
        self.magr.validate()
        if self.weight_quantizer not in _WEIGHT_QUANTIZERS:
            raise ValueError(
                f"weight_quantizer must be one of {_WEIGHT_QUANTIZERS}, got {self.weight_quantizer!r}"
            )
        self.gptq.validate()
        if self.missing_policy not in _MISSING_POLICIES:
            raise ValueError(
                f"missing_policy must be one of {_MISSING_POLICIES}, got {self.missing_policy!r}"
            )
        invalid = sorted(set(self.experts) - set(WAN_EXPERTS))
        if invalid:
            raise ValueError(f"unsupported Wan experts: {invalid}")
        if self.include_regex:
            re.compile(self.include_regex)
        if self.exclude_regex:
            re.compile(self.exclude_regex)

    @property
    def use_smooth(self) -> bool:
        return self.preprocess in {"smooth", "smooth_magr"}

    @property
    def use_magr(self) -> bool:
        return self.preprocess in {"magr", "smooth_magr"}


@dataclass
class WanSmoothedWeightRecord:
    weight: torch.Tensor
    scale: torch.Tensor
    input_channels_dim: int = -1

    def validate(self, *, in_features: int, out_features: int, module_name: str) -> None:
        if tuple(self.weight.shape) != (out_features, in_features):
            raise ValueError(
                f"{module_name}: smoothed weight shape={tuple(self.weight.shape)}, "
                f"expected={(out_features, in_features)}"
            )
        if self.scale.ndim != 1 or int(self.scale.numel()) != int(in_features):
            raise ValueError(
                f"{module_name}: smooth scale shape={tuple(self.scale.shape)}, "
                f"expected=({in_features},)"
            )
        if self.input_channels_dim not in {-1, 1, 2, 3, 4}:
            # Wan DiT Linear inputs normally use the last dimension.  Keep the
            # check permissive enough for captured batched tensors.
            raise ValueError(
                f"{module_name}: unsupported input_channels_dim={self.input_channels_dim}"
            )
        if not torch.isfinite(self.weight).all() or not torch.isfinite(self.scale).all():
            raise ValueError(f"{module_name}: smoothed artifact contains NaN/Inf")


@dataclass
class WanNativeModuleState:
    module_name: str
    weight_qdq: torch.Tensor
    bias: torch.Tensor | None
    smooth_scale: torch.Tensor | None
    algorithm: str
    transforms: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
    report: dict[str, Any] = field(default_factory=dict)


@lru_cache(maxsize=1)
def _load_wan_native_backend() -> Any:
    """Resolve the backend installed in the user's Wan repository.

    The previously validated Wan integration exposes ``quant.hif4_backend``.
    Keeping this import lazy lets the generic package remain importable outside
    the Wan repository.
    """

    try:
        return importlib.import_module("quant.hif4_backend")
    except Exception as exc:  # pragma: no cover - depends on external Wan repo
        raise ImportError(
            "Cannot import Wan native HiF4 backend 'quant.hif4_backend'. "
            "Run from the Wan repository and include both Wan and HiF4 roots in PYTHONPATH."
        ) from exc


def _call_supported(func: Any, tensor: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Call a backend function while tolerating older wrapper signatures."""

    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        signature = None

    if signature is not None:
        accepted = {
            key: value
            for key, value in kwargs.items()
            if key in signature.parameters
        }
        out = func(tensor, **accepted)
    else:  # pragma: no cover - extension functions may not expose signatures
        last_error: Exception | None = None
        candidates = [
            kwargs,
            {k: v for k, v in kwargs.items() if k != "dim"},
            {k: v for k, v in kwargs.items() if k != "force_fp32"},
            {},
        ]
        out = None
        for candidate in candidates:
            try:
                out = func(tensor, **candidate)
                break
            except TypeError as exc:
                last_error = exc
        if out is None:
            assert last_error is not None
            raise last_error

    if not torch.is_tensor(out):
        raise TypeError(f"Wan HiF4 backend returned {type(out).__name__}, expected Tensor")
    return out


def native_hif4_weight_qdq(
    weight: torch.Tensor,
    *,
    force_fp32: bool = True,
) -> torch.Tensor:
    backend = _load_wan_native_backend()
    func = getattr(backend, "hif4_rtn_qdq_weight", None)
    if func is None:
        func = getattr(backend, "hif4_qdq_tensor", None)
    if func is None:
        raise AttributeError(
            "quant.hif4_backend must expose hif4_rtn_qdq_weight or hif4_qdq_tensor"
        )
    out = _call_supported(func, weight, dim=-1, force_fp32=force_fp32)
    _RUNTIME_COUNTERS["weight_qdq_calls"] += 1
    return out


def native_hif4_activation_qdq(
    activation: torch.Tensor,
    *,
    force_fp32: bool = True,
) -> torch.Tensor:
    backend = _load_wan_native_backend()
    func = getattr(backend, "hif4_qdq_tensor", None)
    if func is None:
        raise AttributeError("quant.hif4_backend must expose hif4_qdq_tensor")
    out = _call_supported(func, activation, dim=-1, force_fp32=force_fp32)
    _RUNTIME_COUNTERS["activation_qdq_calls"] += 1
    return out


class WanPreparedHiF4Linear(nn.Module):
    """Wan-compatible native HiF4 fake-QDQ Linear with Smooth compensation."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        weight_qdq: torch.Tensor,
        bias: torch.Tensor | None,
        smooth_scale: torch.Tensor | None,
        activation_qdq: bool,
        force_fp32: bool,
        source_name: str,
        algorithm: str,
        transforms: Sequence[str],
    ) -> None:
        super().__init__()
        if tuple(weight_qdq.shape) != (out_features, in_features):
            raise ValueError(
                f"{source_name}: weight shape={tuple(weight_qdq.shape)}, "
                f"expected={(out_features, in_features)}"
            )
        if smooth_scale is not None and (
            smooth_scale.ndim != 1 or int(smooth_scale.numel()) != int(in_features)
        ):
            raise ValueError(f"{source_name}: smooth_scale shape mismatch")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.activation_qdq = bool(activation_qdq)
        self.force_fp32 = bool(force_fp32)
        self.source_name = str(source_name)
        self.algorithm = str(algorithm)
        self.transforms = tuple(str(item) for item in transforms)

        self.weight = nn.Parameter(weight_qdq.detach().contiguous(), requires_grad=False)
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(bias.detach().contiguous(), requires_grad=False)

        if smooth_scale is None:
            self.register_buffer("smooth_scale", None, persistent=True)
        else:
            self.register_buffer(
                "smooth_scale",
                smooth_scale.detach().cpu().float().contiguous(),
                persistent=True,
            )

    @classmethod
    def from_native_state(
        cls,
        original: nn.Linear,
        state: WanNativeModuleState,
        *,
        activation_qdq: bool,
        force_fp32: bool,
    ) -> "WanPreparedHiF4Linear":
        state_bias = state.bias
        if state_bias is None and original.bias is not None:
            state_bias = original.bias.detach()
        return cls(
            original.in_features,
            original.out_features,
            weight_qdq=state.weight_qdq.to(
                device=original.weight.device,
                dtype=original.weight.dtype,
            ),
            bias=(
                None
                if state_bias is None
                else state_bias.to(
                    device=original.weight.device,
                    dtype=original.weight.dtype,
                )
            ),
            smooth_scale=state.smooth_scale,
            activation_qdq=activation_qdq,
            force_fp32=force_fp32,
            source_name=state.module_name,
            algorithm=state.algorithm,
            transforms=state.transforms,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _RUNTIME_COUNTERS["linear_forward_calls"] += 1
        work = x
        if self.smooth_scale is not None:
            # Smooth compensation must precede activation quantization.
            scale = self.smooth_scale.to(device=x.device, dtype=torch.float32)
            view_shape = [1] * x.ndim
            view_shape[-1] = int(scale.numel())
            work = (work.float() / scale.view(*view_shape)).to(dtype=x.dtype)
        if self.activation_qdq:
            work = native_hif4_activation_qdq(
                work,
                force_fp32=self.force_fp32,
            ).to(dtype=x.dtype)
        return F.linear(work, self.weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"algorithm={self.algorithm!r}, transforms={list(self.transforms)!r}, "
            f"activation_qdq={self.activation_qdq}, "
            f"smooth={self.smooth_scale is not None}, source_name={self.source_name!r}"
        )


def _torch_load_dict(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"artifact {path} must contain a dict, got {type(payload).__name__}")
    return payload


def load_wan_activation_stats(
    path: str | Path,
) -> dict[str, dict[str, ActivationStats]]:
    """Load Wan ``act.pt`` records (``s``) into generic ActivationStats."""

    payload = _torch_load_dict(path)
    output: dict[str, dict[str, ActivationStats]] = {}
    for expert in WAN_EXPERTS:
        records = payload.get(expert, {})
        if not isinstance(records, Mapping):
            raise TypeError(f"act artifact {expert} must be a mapping")
        expert_out: dict[str, ActivationStats] = {}
        for name, record in records.items():
            if not isinstance(record, Mapping):
                continue
            values = record.get("s", record.get("absmax"))
            if not torch.is_tensor(values):
                continue
            stats = ActivationStats(
                absmax=values.detach().cpu().float().contiguous(),
                num_calls=max(1, int(record.get("num_calls", 1))),
                channels_dim=int(record.get("channels_dim", -1)),
                module_name=str(name),
            )
            stats.validate()
            expert_out[str(name)] = stats
        output[expert] = expert_out
    return output


def load_wan_curvature_stats(
    path: str | Path,
    *,
    default_normalization: str = "mean_xtx",
) -> dict[str, dict[str, CurvatureStats]]:
    """Load Wan GPTQ Hessian artifacts.

    The public Wan collector's ``export(normalize=True)`` record stores
    ``hessian`` as mean-XtX, hence the default normalization is ``mean_xtx``.
    Explicit artifact metadata always takes precedence.
    """

    payload = _torch_load_dict(path)
    output: dict[str, dict[str, CurvatureStats]] = {}
    for expert in WAN_EXPERTS:
        records = payload.get(expert, {})
        if not isinstance(records, Mapping):
            raise TypeError(f"Hessian artifact {expert} must be a mapping")
        expert_out: dict[str, CurvatureStats] = {}
        for name, record in records.items():
            if not isinstance(record, Mapping):
                continue
            matrix = record.get("hessian", record.get("matrix"))
            if not torch.is_tensor(matrix):
                continue
            in_features = int(record.get("in_features", matrix.shape[-1]))
            normalization = str(record.get("normalization", default_normalization))
            stats = CurvatureStats(
                matrix=matrix.detach().cpu().float().contiguous(),
                nsamples=max(1, int(record.get("nsamples", 1))),
                in_features=in_features,
                normalization=normalization,
                domain=str(record.get("domain", "original")),
                module_name=str(name),
            )
            stats.validate()
            expert_out[str(name)] = stats
        output[expert] = expert_out
    return output


def load_wan_smoothed_weights(
    path: str | Path,
) -> dict[str, dict[str, WanSmoothedWeightRecord]]:
    """Load the Wan Smooth pipeline's ``wgt.pt`` directly."""

    payload = _torch_load_dict(path)
    output: dict[str, dict[str, WanSmoothedWeightRecord]] = {}
    for expert in WAN_EXPERTS:
        records = payload.get(expert, {})
        if not isinstance(records, Mapping):
            raise TypeError(f"wgt artifact {expert} must be a mapping")
        expert_out: dict[str, WanSmoothedWeightRecord] = {}
        for name, record in records.items():
            if not isinstance(record, Mapping):
                continue
            weight = record.get("weight")
            scale = record.get("scale", record.get("smooth_scale"))
            if not torch.is_tensor(weight) or not torch.is_tensor(scale):
                continue
            expert_out[str(name)] = WanSmoothedWeightRecord(
                weight=weight.detach().cpu().contiguous(),
                scale=scale.detach().cpu().float().contiguous(),
                input_channels_dim=int(record.get("input_channels_dim", -1)),
            )
        output[expert] = expert_out
    return output


def select_wan_hif4_linears(
    model: nn.Module,
    *,
    include_regex: str = "",
    exclude_regex: str = "",
    explicit_names: Iterable[str] | None = None,
) -> list[str]:
    """Select the same Wan DiT block Linear scope used by the HiF4 baseline."""

    if explicit_names is not None:
        requested = sorted(set(str(item) for item in explicit_names))
        for name in requested:
            if not isinstance(get_module(model, name), nn.Linear):
                raise TypeError(f"explicit target {name!r} is not nn.Linear")
        return requested

    include = re.compile(include_regex) if include_regex else None
    exclude = re.compile(exclude_regex) if exclude_regex else None
    names: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not name.startswith("blocks."):
            continue
        lowered = name.lower()
        if any(token in lowered for token in _DEFAULT_SKIP_TOKENS):
            continue
        if include is not None and include.search(name) is None:
            continue
        if exclude is not None and exclude.search(name) is not None:
            continue
        names.append(name)
    return sorted(names)


def _resolve_quant_device(requested: str, fallback: torch.device) -> torch.device:
    value = str(requested).strip()
    if not value:
        return fallback
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("quant_device='cuda' requested but CUDA is unavailable")
        return torch.device("cuda", torch.cuda.current_device())
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"quant_device={value!r} requested but CUDA is unavailable")
    return device


def _prepare_from_wan_smoothed_record(
    original_weight: torch.Tensor,
    record: WanSmoothedWeightRecord,
    *,
    curvature: CurvatureStats | None,
    magr_config: MagRConfig | None,
    module_name: str,
) -> PreparedWeightState:
    record.validate(
        in_features=int(original_weight.shape[1]),
        out_features=int(original_weight.shape[0]),
        module_name=module_name,
    )
    weight = record.weight.to(
        device=original_weight.device,
        dtype=original_weight.dtype,
    )
    scale = record.scale.detach().cpu().float().contiguous()
    transformed_curvature = curvature
    if curvature is not None and curvature.domain != "smoothed":
        transformed_curvature = transform_curvature_for_smooth(curvature, scale)

    transforms = ["smoothquant_wan_wgt"]
    metadata: dict[str, Any] = {
        "transform": "smoothquant",
        "source": "wan_wgt_artifact",
        "input_compensation": "x / smooth_scale before activation QDQ",
    }
    report: dict[str, Any] = {
        "smooth_scale_min": float(scale.min().item()),
        "smooth_scale_max": float(scale.max().item()),
        "smooth_scale_mean": float(scale.mean().item()),
    }
    reference = None
    if magr_config is not None:
        if transformed_curvature is None:
            raise ValueError(f"{module_name}: MagR requires curvature stats")
        reference = weight.detach().clone()
        result: MagRResult = apply_magr(weight, transformed_curvature, magr_config)
        weight = result.weight
        transforms.append(f"magr_{magr_config.mode}")
        metadata.update(result.metadata)
        report.update(result.report)

    return PreparedWeightState(
        weight=weight,
        curvature=transformed_curvature,
        smooth_scale=scale,
        transforms=transforms,
        metadata=metadata,
        report=report,
        quantization_reference_weight=reference,
    )


def _build_native_module_state(
    module: nn.Linear,
    module_name: str,
    *,
    activation: ActivationStats | None,
    curvature: CurvatureStats | None,
    smoothed_record: WanSmoothedWeightRecord | None,
    config: WanHiF4PreprocessConfig,
) -> WanNativeModuleState:
    smooth_config = config.smooth if config.use_smooth else None
    magr_config = config.magr if config.use_magr else None

    if smoothed_record is not None:
        prepared = _prepare_from_wan_smoothed_record(
            module.weight.detach(),
            smoothed_record,
            curvature=curvature,
            magr_config=magr_config,
            module_name=module_name,
        )
    else:
        prepared = prepare_weight_transforms(
            module.weight.detach(),
            activation_stats=activation,
            curvature=curvature,
            smooth_config=smooth_config,
            magr_config=magr_config,
        )

    quant_device = _resolve_quant_device(config.quant_device, module.weight.device)
    quant_metadata: dict[str, Any] = {}
    quant_report: dict[str, Any] = {}
    if config.weight_quantizer == "rtn":
        work_weight = prepared.weight.detach().to(device=quant_device)
        qdq = native_hif4_weight_qdq(
            work_weight,
            force_fp32=config.force_fp32,
        )
        qdq = qdq.to(device=module.weight.device, dtype=module.weight.dtype).contiguous()
        algorithm = "native_hif4_rtn"
        quant_metadata = {
            "native_backend": "quant.hif4_backend",
            "quant_device": str(quant_device),
        }
    elif config.weight_quantizer == "gptq":
        if prepared.curvature is None:
            raise ValueError(f"{module_name}: HiF4 GPTQ requires curvature stats")
        backend = HIFX4Format(force_fp32=config.force_fp32)
        qstate = quantize_gptq_original(
            prepared.weight.detach(),
            prepared.curvature,
            backend,
            config.gptq,
        )
        qdq = qstate.dequantized_weight.to(
            device=module.weight.device, dtype=module.weight.dtype
        ).contiguous()
        algorithm = "hif4_gptq_original"
        quant_metadata = dict(qstate.metadata)
        quant_report = dict(qstate.report)
        _RUNTIME_COUNTERS["weight_qdq_calls"] += 1
    else:  # pragma: no cover - validated above
        raise ValueError(config.weight_quantizer)

    return WanNativeModuleState(
        module_name=module_name,
        weight_qdq=qdq,
        bias=None if module.bias is None else module.bias.detach().cpu().contiguous(),
        smooth_scale=prepared.smooth_scale,
        algorithm=algorithm,
        transforms=list(prepared.transforms),
        metadata={
            **prepared.metadata,
            **quant_metadata,
            "format": "hifx4",
            "algorithm": algorithm,
            "weight_quantizer": config.weight_quantizer,
        },
        report={**prepared.report, **quant_report},
    )


def replace_wan_expert_with_preprocessed_hif4(
    model: nn.Module,
    *,
    expert_name: str,
    config: WanHiF4PreprocessConfig,
    activation_stats: Mapping[str, ActivationStats] | None = None,
    curvature_stats: Mapping[str, CurvatureStats] | None = None,
    smoothed_weights: Mapping[str, WanSmoothedWeightRecord] | None = None,
    module_names: Sequence[str] | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    config.validate()
    log = logger or LOGGER
    selected = select_wan_hif4_linears(
        model,
        include_regex=config.include_regex,
        exclude_regex=config.exclude_regex,
        explicit_names=module_names,
    )
    activation_stats = activation_stats or {}
    curvature_stats = curvature_stats or {}
    smoothed_weights = smoothed_weights or {}

    replaced: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    for index, name in enumerate(selected, start=1):
        module = get_module(model, name)
        if not isinstance(module, nn.Linear):
            raise TypeError(f"{expert_name}.{name}: target is not nn.Linear")

        activation = activation_stats.get(name)
        curvature = curvature_stats.get(name)
        smoothed = smoothed_weights.get(name)

        required_missing: list[str] = []
        if config.use_smooth and smoothed is None and activation is None:
            required_missing.append("activation_or_wgt")
        if (config.use_magr or config.weight_quantizer == "gptq") and curvature is None:
            required_missing.append("curvature")
        fallback_kind = ""
        if required_missing:
            missing.append({"name": name, "missing": ",".join(required_missing)})
            policy = "error" if config.strict_artifacts else config.missing_policy
            if policy == "error":
                raise KeyError(
                    f"{expert_name}.{name}: missing required artifacts {required_missing}"
                )
            if policy == "bf16":
                continue
            if policy != "hif4_rtn":
                raise ValueError(f"unsupported missing policy: {policy}")
            smooth_available = config.use_smooth and (
                smoothed is not None or activation is not None
            )
            magr_available = config.use_magr and curvature is not None
            if smooth_available and magr_available:
                fallback_preprocess = "smooth_magr"
            elif smooth_available:
                fallback_preprocess = "smooth"
            elif magr_available:
                fallback_preprocess = "magr"
            else:
                fallback_preprocess = "none"
            fallback_kind = (
                "native_hif4_rtn"
                if fallback_preprocess == "none"
                else f"{fallback_preprocess}_then_native_hif4_rtn"
            )
            fallback_config = replace(
                config,
                preprocess=fallback_preprocess,
                weight_quantizer="rtn",
                strict_artifacts=False,
                missing_policy="bf16",
            )
            state = _build_native_module_state(
                module,
                name,
                activation=activation if smooth_available else None,
                curvature=curvature if magr_available else None,
                smoothed_record=smoothed if smooth_available else None,
                config=fallback_config,
            )
            state.transforms.append(f"fallback_for_{'_'.join(required_missing)}")
            state.metadata["fallback_reason"] = list(required_missing)
            state.metadata["fallback_preprocess"] = fallback_preprocess
        else:
            state = _build_native_module_state(
                module,
                name,
                activation=activation,
                curvature=curvature,
                smoothed_record=smoothed if config.use_smooth else None,
                config=config,
            )
        replacement = WanPreparedHiF4Linear.from_native_state(
            module,
            state,
            activation_qdq=config.activation_qdq,
            force_fp32=config.force_fp32,
        )
        set_module(model, name, replacement)
        replaced.append(
            {
                "name": name,
                "algorithm": state.algorithm,
                "transforms": list(state.transforms),
                "smooth": state.smooth_scale is not None,
                "fallback": fallback_kind,
                "metadata": state.metadata,
                "report": state.report,
            }
        )
        if index == 1 or index % 25 == 0 or index == len(selected):
            log.info(
                "[%s] native HiF4 preprocess progress %d/%d | replaced=%d",
                expert_name,
                index,
                len(selected),
                len(replaced),
            )
        del state, replacement
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report = {
        "expert": expert_name,
        "format": "hifx4",
        "algorithm": (
            "native_hif4_rtn" if config.weight_quantizer == "rtn"
            else "hif4_gptq_original"
        ),
        "weight_quantizer": config.weight_quantizer,
        "preprocess": config.preprocess,
        "activation_qdq": bool(config.activation_qdq),
        "selected_count": len(selected),
        "replaced_count": len(replaced),
        "missing_count": len(missing),
        "fallback_count": sum(1 for item in replaced if item.get("fallback")),
        "fallback_hif4_count": sum(1 for item in replaced if item.get("fallback") == "native_hif4_rtn"),
        "fallback_smooth_count": sum(1 for item in replaced if str(item.get("fallback", "")).startswith("smooth_then")),
        "kept_bf16_count": len(selected) - len(replaced),
        "missing_policy": config.missing_policy,
        "missing": missing,
        "modules": replaced,
    }
    log.info(
        "[%s] native HiF4 preprocess finished | selected=%d replaced=%d missing=%d preprocess=%s",
        expert_name,
        len(selected),
        len(replaced),
        len(missing),
        config.preprocess,
    )
    return report



def _export_wan_preprocessed_cache(pipe: Any, payload_meta: Mapping[str, Any]) -> dict[str, Any]:
    cache: dict[str, Any] = {
        "__meta__": dict(payload_meta),
        "low_noise_model": {},
        "high_noise_model": {},
    }
    cache["__meta__"]["artifact"] = "wan_hif4_preprocessed_linear_cache"
    for expert in WAN_EXPERTS:
        model = getattr(pipe, expert)
        for name, module in model.named_modules():
            if not isinstance(module, WanPreparedHiF4Linear):
                continue
            cache[expert][name] = {
                "in_features": module.in_features,
                "out_features": module.out_features,
                "weight_qdq": module.weight.detach().cpu().contiguous(),
                "bias": None if module.bias is None else module.bias.detach().cpu().contiguous(),
                "smooth_scale": None if module.smooth_scale is None else module.smooth_scale.detach().cpu().contiguous(),
                "activation_qdq": bool(module.activation_qdq),
                "force_fp32": bool(module.force_fp32),
                "algorithm": module.algorithm,
                "transforms": list(module.transforms),
                "source_name": module.source_name,
            }
    cache["__meta__"]["module_counts"] = {
        expert: len(cache[expert]) for expert in WAN_EXPERTS
    }
    return cache


def _save_wan_preprocessed_cache(path: str | Path, cache: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    torch.save(dict(cache), tmp)
    tmp.replace(target)


def _load_wan_preprocessed_cache_into_pipe(
    pipe: Any,
    path: str | Path,
    *,
    logger: logging.Logger,
    activation_qdq_override: bool | None = None,
) -> dict[str, Any]:
    cache = _torch_load_dict(path)
    reports: dict[str, Any] = {}
    for expert in WAN_EXPERTS:
        model = getattr(pipe, expert, None)
        if not isinstance(model, nn.Module):
            raise AttributeError(f"Wan pipeline missing expert {expert}")
        records = cache.get(expert, {})
        if not isinstance(records, Mapping):
            raise TypeError(f"cache[{expert}] must be a mapping")
        replaced = 0
        for name, record in records.items():
            original = get_module(model, str(name))
            if not isinstance(original, nn.Linear):
                raise TypeError(
                    f"cache target {expert}.{name} expected original nn.Linear, got {type(original).__name__}"
                )
            weight = record.get("weight_qdq")
            if not torch.is_tensor(weight):
                raise TypeError(f"cache target {expert}.{name} missing weight_qdq")
            replacement = WanPreparedHiF4Linear(
                int(record.get("in_features", original.in_features)),
                int(record.get("out_features", original.out_features)),
                weight_qdq=weight.to(device=original.weight.device, dtype=original.weight.dtype),
                bias=(
                    original.bias.detach().to(
                        device=original.weight.device, dtype=original.weight.dtype
                    )
                    if "bias" not in record and original.bias is not None
                    else (
                        None
                        if record.get("bias") is None
                        else record["bias"].to(
                            device=original.weight.device, dtype=original.weight.dtype
                        )
                    )
                ),
                smooth_scale=record.get("smooth_scale"),
                activation_qdq=(
                    bool(record.get("activation_qdq", True))
                    if activation_qdq_override is None
                    else bool(activation_qdq_override)
                ),
                force_fp32=bool(record.get("force_fp32", True)),
                source_name=str(record.get("source_name", name)),
                algorithm=str(record.get("algorithm", "native_hif4_rtn")),
                transforms=list(record.get("transforms", [])),
            )
            set_module(model, str(name), replacement)
            replaced += 1
        reports[expert] = {
            "expert": expert,
            "cache_loaded": True,
            "selected_count": replaced,
            "replaced_count": replaced,
            "missing_count": 0,
            "fallback_count": 0,
            "modules": list(records.keys()),
        }
        logger.info("[%s] loaded %d preprocessed HiF4 Linear modules from cache", expert, replaced)
    return {
        "__meta__": {
            **dict(cache.get("__meta__", {})),
            "cache_loaded": True,
            "cache_path": str(path),
            "activation_qdq_override": activation_qdq_override,
        },
        **reports,
    }


def setup_wan_i2v_hif4_preprocess(
    pipe: Any,
    *,
    config: WanHiF4PreprocessConfig,
    act_path: str | Path | None = None,
    wgt_path: str | Path | None = None,
    hessian_path: str | Path | None = None,
    hessian_default_normalization: str = "mean_xtx",
    report_json: str | Path | None = None,
    cache_path: str | Path | None = None,
    rebuild_cache: bool = False,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Replace selected Linear modules in both Wan I2V experts.

    ``wgt_path`` takes precedence for Smooth because it is the exact output of
    the existing Wan Smooth pipeline.  ``act_path`` is used to recompute scales
    when ``wgt.pt`` is unavailable.
    """

    config.validate()
    log = logger or LOGGER
    cache_target = None if cache_path is None or not str(cache_path).strip() else Path(cache_path)
    if cache_target is not None and cache_target.is_file() and not rebuild_cache:
        payload = _load_wan_preprocessed_cache_into_pipe(
            pipe,
            cache_target,
            logger=log,
            # Tri-state semantics: default preserves cache metadata; the explicit
            # --hif4_no_activation_qdq flag forces False at runtime.
            activation_qdq_override=(False if not config.activation_qdq else None),
        )
        if report_json is not None and str(report_json).strip():
            target = Path(report_json)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return payload

    activation = (
        load_wan_activation_stats(act_path)
        if act_path is not None and str(act_path).strip()
        else {expert: {} for expert in WAN_EXPERTS}
    )
    curvature = (
        load_wan_curvature_stats(
            hessian_path,
            default_normalization=hessian_default_normalization,
        )
        if hessian_path is not None and str(hessian_path).strip()
        else {expert: {} for expert in WAN_EXPERTS}
    )
    smoothed = (
        load_wan_smoothed_weights(wgt_path)
        if wgt_path is not None and str(wgt_path).strip()
        else {expert: {} for expert in WAN_EXPERTS}
    )

    reports: dict[str, Any] = {}
    for expert in config.experts:
        model = getattr(pipe, expert, None)
        if not isinstance(model, nn.Module):
            raise AttributeError(f"Wan pipeline does not expose nn.Module {expert!r}")
        reports[expert] = replace_wan_expert_with_preprocessed_hif4(
            model,
            expert_name=expert,
            config=config,
            activation_stats=activation.get(expert),
            curvature_stats=curvature.get(expert),
            smoothed_weights=smoothed.get(expert),
            logger=log,
        )

    payload = {
        "__meta__": {
            "adapter": "HiFloat4.ptq.adapters.wan_i2v",
            "format": "hifx4",
            "algorithm": (
                "native_hif4_rtn" if config.weight_quantizer == "rtn"
                else "hif4_gptq_original"
            ),
            "weight_quantizer": config.weight_quantizer,
            "preprocess": config.preprocess,
            "act_path": "" if act_path is None else str(act_path),
            "wgt_path": "" if wgt_path is None else str(wgt_path),
            "hessian_path": "" if hessian_path is None else str(hessian_path),
            "hessian_default_normalization": hessian_default_normalization,
            "config": {
                "activation_qdq": config.activation_qdq,
                "force_fp32": config.force_fp32,
                "include_regex": config.include_regex,
                "exclude_regex": config.exclude_regex,
                "quant_device": config.quant_device,
                "weight_quantizer": config.weight_quantizer,
                "gptq": asdict(config.gptq),
                "strict_artifacts": config.strict_artifacts,
                "missing_policy": config.missing_policy,
                "experts": list(config.experts),
                "smooth": asdict(config.smooth),
                "magr": asdict(config.magr),
            },
        },
        **reports,
    }
    if cache_target is not None:
        cache_payload = _export_wan_preprocessed_cache(pipe, payload["__meta__"])
        _save_wan_preprocessed_cache(cache_target, cache_payload)
        payload["__meta__"]["cache_saved"] = True
        payload["__meta__"]["cache_path"] = str(cache_target)
        payload["__meta__"]["cache_bytes"] = int(cache_target.stat().st_size)
        log.info("Wan HiF4 preprocessed cache written to %s", cache_target)

    if report_json is not None and str(report_json).strip():
        target = Path(report_json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info("Wan HiF4 preprocess report written to %s", target)
    return payload


def add_wan_hif4_preprocess_args(parser: argparse.ArgumentParser) -> None:
    """Add non-conflicting arguments to ``generate_vbench_i2v_svdquant.py``."""

    group = parser.add_argument_group("Wan native HiF4 Smooth/MagR adapter")
    group.add_argument(
        "--hif4_preprocess",
        choices=list(_PREPROCESS_CHOICES),
        default="none",
        help="Preprocess before the selected HiF4 RTN/GPTQ weight quantizer.",
    )
    group.add_argument(
        "--hif4_weight_quantizer",
        choices=list(_WEIGHT_QUANTIZERS),
        default="rtn",
        help="HiF4 weight quantizer after Smooth/MagR: native RTN or original GPTQ.",
    )
    group.add_argument("--hif4_gptq_block_size", type=int, default=64)
    group.add_argument("--hif4_gptq_damp_percent", type=float, default=0.01)
    group.add_argument("--hif4_gptq_compute_device", default="cuda")
    group.add_argument(
        "--hif4_wan_act_path",
        default="",
        help="Wan Smooth act.pt; used when wgt.pt is not supplied.",
    )
    group.add_argument(
        "--hif4_wan_wgt_path",
        default="",
        help="Wan Smooth wgt.pt. Preferred exact bridge for Smooth -> HiF4.",
    )
    group.add_argument(
        "--hif4_wan_hessian_path",
        default="",
        help="Wan GPTQ Hessian artifact for MagR.",
    )
    group.add_argument(
        "--hif4_hessian_normalization",
        choices=["mean_xtx", "sum_xtx"],
        default="mean_xtx",
    )
    group.add_argument("--hif4_smooth_alpha", type=float, default=0.5)
    group.add_argument("--hif4_smooth_eps", type=float, default=1e-5)
    group.add_argument(
        "--hif4_magr_mode",
        choices=["group64", "leaf4", "hif4_tree"],
        default="group64",
    )
    group.add_argument("--hif4_magr_alpha", type=float, default=1e-4)
    group.add_argument("--hif4_magr_iters", type=int, default=200)
    group.add_argument(
        "--hif4_magr_spectral",
        choices=["power", "exact"],
        default="exact",
    )
    group.add_argument("--hif4_magr_power_iters", type=int, default=32)
    group.add_argument("--hif4_magr_dykstra_iters", type=int, default=32)
    group.add_argument("--hif4_magr_compute_device", default="cuda")
    group.add_argument("--hif4_preprocess_quant_device", default="cuda")
    group.add_argument("--hif4_preprocess_report_json", default="")
    group.add_argument("--hif4_preprocess_runtime_report_json", default="")
    group.add_argument("--hif4_preprocess_cache_path", default="")
    group.add_argument("--hif4_preprocess_rebuild_cache", action="store_true", default=False)
    group.add_argument(
        "--hif4_preprocess_missing_policy",
        choices=list(_MISSING_POLICIES),
        default="error",
        help="Missing transform artifacts: fail, use ordinary HiF4 RTN, or keep BF16.",
    )
    group.add_argument(
        "--hif4_preprocess_allow_missing",
        action="store_true",
        default=False,
        help="Skip modules missing calibration instead of failing closed.",
    )


def wan_hif4_config_from_args(args: argparse.Namespace) -> WanHiF4PreprocessConfig:
    return WanHiF4PreprocessConfig(
        preprocess=str(getattr(args, "hif4_preprocess", "none")),
        smooth=SmoothConfig(
            alpha=float(getattr(args, "hif4_smooth_alpha", 0.5)),
            eps=float(getattr(args, "hif4_smooth_eps", 1e-5)),
        ),
        magr=MagRConfig(
            mode=str(getattr(args, "hif4_magr_mode", "group64")),
            alpha=float(getattr(args, "hif4_magr_alpha", 1e-4)),
            outer_iters=int(getattr(args, "hif4_magr_iters", 200)),
            spectral_method=str(getattr(args, "hif4_magr_spectral", "exact")),
            power_iters=int(getattr(args, "hif4_magr_power_iters", 32)),
            dykstra_iters=int(getattr(args, "hif4_magr_dykstra_iters", 32)),
            compute_device=str(getattr(args, "hif4_magr_compute_device", "cuda")),
        ),
        weight_quantizer=str(getattr(args, "hif4_weight_quantizer", "rtn")),
        gptq=OriginalGPTQConfig(
            block_size=int(getattr(args, "hif4_gptq_block_size", 64)),
            damp_percent=float(getattr(args, "hif4_gptq_damp_percent", 0.01)),
            compute_device=str(getattr(args, "hif4_gptq_compute_device", "cuda")),
        ),
        activation_qdq=not bool(getattr(args, "hif4_no_activation_qdq", False)),
        force_fp32=bool(getattr(args, "hif4_force_fp32", True)),
        include_regex=str(getattr(args, "hif4_include_regex", "")),
        exclude_regex=str(getattr(args, "hif4_exclude_regex", "")),
        quant_device=str(getattr(args, "hif4_preprocess_quant_device", "cuda")),
        strict_artifacts=(
            str(getattr(args, "hif4_preprocess_missing_policy", "error")) == "error"
            and not bool(getattr(args, "hif4_preprocess_allow_missing", False))
        ),
        missing_policy=(
            "hif4_rtn"
            if bool(getattr(args, "hif4_preprocess_allow_missing", False))
            else str(getattr(args, "hif4_preprocess_missing_policy", "error"))
        ),
    )


def setup_wan_hif4_preprocess_from_args(
    pipe: Any,
    args: argparse.Namespace,
    *,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Generator bridge.

    Returns ``{"handled": False}`` for non-HiF4 mode or ``preprocess=none``.
    The caller should then execute the already validated legacy
    ``_setup_hif4_if_needed``.  When ``handled`` is true, this adapter has
    performed the replacement and the legacy setup must be skipped to avoid
    double QDQ/replacement.
    """

    if str(getattr(args, "mode", "")) != "hif4":
        return {"handled": False, "reason": "mode_is_not_hif4"}
    config = wan_hif4_config_from_args(args)
    if config.preprocess == "none":
        return {"handled": False, "reason": "preprocess_none"}

    reset_wan_hif4_runtime_counters()
    runtime_report_path = str(
        getattr(args, "hif4_preprocess_runtime_report_json", "")
    ).strip()
    if runtime_report_path:
        _RUNTIME_REPORT_CONTEXT.clear()
        _RUNTIME_REPORT_CONTEXT.update(
            {
                "preprocess": config.preprocess,
                "runtime_report_path": runtime_report_path,
            }
        )
        atexit.register(write_wan_hif4_runtime_report, runtime_report_path)

    report = setup_wan_i2v_hif4_preprocess(
        pipe,
        config=config,
        act_path=str(getattr(args, "hif4_wan_act_path", "")) or None,
        wgt_path=str(getattr(args, "hif4_wan_wgt_path", "")) or None,
        hessian_path=str(getattr(args, "hif4_wan_hessian_path", "")) or None,
        hessian_default_normalization=str(
            getattr(args, "hif4_hessian_normalization", "mean_xtx")
        ),
        report_json=str(getattr(args, "hif4_preprocess_report_json", "")) or None,
        cache_path=str(getattr(args, "hif4_preprocess_cache_path", "")) or None,
        rebuild_cache=bool(getattr(args, "hif4_preprocess_rebuild_cache", False)),
        logger=logger,
    )
    return {"handled": True, "report": report}


__all__ = [
    "WAN_EXPERTS",
    "WanHiF4PreprocessConfig",
    "WanNativeModuleState",
    "WanPreparedHiF4Linear",
    "WanSmoothedWeightRecord",
    "add_wan_hif4_preprocess_args",
    "load_wan_activation_stats",
    "load_wan_curvature_stats",
    "load_wan_smoothed_weights",
    "native_hif4_activation_qdq",
    "native_hif4_weight_qdq",
    "reset_wan_hif4_runtime_counters",
    "get_wan_hif4_runtime_counters",
    "write_wan_hif4_runtime_report",
    "replace_wan_expert_with_preprocessed_hif4",
    "select_wan_hif4_linears",
    "setup_wan_hif4_preprocess_from_args",
    "setup_wan_i2v_hif4_preprocess",
    "wan_hif4_config_from_args",
]
