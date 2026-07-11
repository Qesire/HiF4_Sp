"""把普通 ``nn.Linear`` 替换为 HiF4 W4A4 fake-quant Linear。"""

from __future__ import annotations

import re
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hif4_backend import hif4_qdq_tensor, hif4_rtn_qdq_weight


class HiF4FakeQuantLinear(nn.Module):
    """HiF4 W4A4 fake-quant Linear。

    - 权重：构建/替换时执行一次 HiF4 RTN-QDQ；
    - 激活：每次 forward 在线执行 HiF4 QDQ；
    - 计算：反量化后的浮点 ``F.linear``，通常为 BF16 GEMM。
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        weight_qdq: torch.Tensor | None = None,
        bias_tensor: torch.Tensor | None = None,
        activation_qdq: bool = True,
        force_fp32: bool = True,
        source_name: str = "",
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.activation_qdq = bool(activation_qdq)
        self.force_fp32 = bool(force_fp32)
        self.source_name = source_name

        if weight_qdq is None:
            weight_qdq = torch.empty((out_features, in_features), dtype=torch.bfloat16)
        self.weight = nn.Parameter(weight_qdq, requires_grad=False)

        if bias:
            if bias_tensor is None:
                bias_tensor = torch.zeros(
                    out_features,
                    dtype=weight_qdq.dtype,
                    device=weight_qdq.device,
                )
            self.bias = nn.Parameter(bias_tensor, requires_grad=False)
        else:
            self.register_parameter("bias", None)

    @classmethod
    @torch.no_grad()
    def from_linear(
        cls,
        module: nn.Linear,
        *,
        activation_qdq: bool = True,
        force_fp32: bool = True,
        source_name: str = "",
        quant_device: str = "cuda",
    ) -> "HiF4FakeQuantLinear":
        weight = module.weight.detach()
        weight_qdq = hif4_rtn_qdq_weight(
            weight,
            dim=-1,
            force_fp32=force_fp32,
            quant_device=quant_device,
            return_device=weight.device,
        ).to(weight.dtype)
        bias_tensor = None if module.bias is None else module.bias.detach().clone()

        return cls(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            weight_qdq=weight_qdq,
            bias_tensor=bias_tensor,
            activation_qdq=activation_qdq,
            force_fp32=force_fp32,
            source_name=source_name,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_qdq:
            x = hif4_qdq_tensor(
                x,
                dim=-1,
                force_fp32=self.force_fp32,
                kind="activation",
            ).to(x.dtype)

        weight = self.weight.to(device=x.device, dtype=x.dtype)
        bias = None if self.bias is None else self.bias.to(device=x.device, dtype=x.dtype)
        return F.linear(x, weight, bias)


def _get_parent(root: nn.Module, name: str) -> tuple[nn.Module, str]:
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def should_quantize_linear(
    name: str,
    module: nn.Linear,
    *,
    include_regex: str | None = None,
    exclude_regex: str | None = None,
) -> bool:
    """选择 Wan DiT attention/FFN Linear，并排除时间/AdaLN/头部等敏感层。"""

    del module
    lower_name = name.lower()

    if exclude_regex and re.search(exclude_regex, name):
        return False
    if include_regex and not re.search(include_regex, name):
        return False

    default_exclude = r"(time|timestep|embed|embedding|modulation|adaln|head|final|norm|patch|vae|t5|text)"
    default_include = r"(self_attn|cross_attn|attn|ffn|mlp|feed_forward|to_q|to_k|to_v|to_out|proj|q_proj|k_proj|v_proj|o_proj|gate|up|down)"

    if re.search(default_exclude, lower_name):
        return False
    return bool(re.search(default_include, lower_name))


@torch.no_grad()
def replace_linear_with_hif4(
    model: nn.Module,
    *,
    include_regex: str | None = None,
    exclude_regex: str | None = None,
    activation_qdq: bool = True,
    force_fp32: bool = True,
    dry_run: bool = False,
    quant_device: str = "cuda",
) -> dict[str, Any]:
    """扫描模型、记录选择清单，并原位替换目标 Linear。"""

    selected: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        chosen = should_quantize_linear(
            name,
            module,
            include_regex=include_regex,
            exclude_regex=exclude_regex,
        )
        row = {
            "name": name,
            "in_features": module.in_features,
            "out_features": module.out_features,
            "bias": module.bias is not None,
            "dtype": str(module.weight.dtype),
            "device": str(module.weight.device),
            "selected": chosen,
        }
        (selected if chosen else skipped).append(row)

    if not dry_run:
        for row in selected:
            name = str(row["name"])
            parent, child = _get_parent(model, name)
            old = getattr(parent, child)
            new = HiF4FakeQuantLinear.from_linear(
                old,
                activation_qdq=activation_qdq,
                force_fp32=force_fp32,
                source_name=name,
                quant_device=quant_device,
            )
            setattr(parent, child, new)

    return {
        "dry_run": dry_run,
        "selected_count": len(selected),
        "skipped_count": len(skipped),
        "selected": selected,
        "skipped": skipped,
    }
