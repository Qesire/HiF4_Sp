"""Wan2.2-I2V 的 HiF4 RTN-QDQ 适配接口。

该子包只提供 fake-quant 数值基线：
BF16 checkpoint -> HiF4 weight RTN-QDQ -> online activation QDQ -> BF16 F.linear。
它不是 packed 4-bit GEMM，也不承诺吞吐或显存收益。
"""

from .hif4_backend import (
    get_hif4_api,
    get_hif4_stats,
    hif4_backend_metadata,
    hif4_qdq_tensor,
    hif4_rtn_qdq_weight,
    reset_hif4_stats,
)
from .hif4_linear import (
    HiF4FakeQuantLinear,
    replace_linear_with_hif4,
    should_quantize_linear,
)

__all__ = [
    "HiF4FakeQuantLinear",
    "get_hif4_api",
    "get_hif4_stats",
    "hif4_backend_metadata",
    "hif4_qdq_tensor",
    "hif4_rtn_qdq_weight",
    "replace_linear_with_hif4",
    "reset_hif4_stats",
    "should_quantize_linear",
]
