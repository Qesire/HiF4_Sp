from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

from ..state import GroupLayout


class WeightFormatBackend(ABC):
    """RTN/GPTQ 共同消费的格式接口。"""

    name: str
    group_layout: GroupLayout

    @abstractmethod
    def quantize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def quantize_weight(
        self,
        weight: torch.Tensor,
        *,
        quant_device: str | torch.device = "cuda",
        return_device: str | torch.device | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def quantize_activation(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def prepare_group(self, weight_group: torch.Tensor, *, group_start: int = 0) -> Any:
        """根据指定权重 group 生成随后冻结的格式状态。"""
        raise NotImplementedError

    @abstractmethod
    def quantize_column(
        self,
        column: torch.Tensor,
        group_state: Any,
        local_column_index: int,
    ) -> torch.Tensor:
        raise NotImplementedError

    def quantize_column_with_code(
        self,
        column: torch.Tensor,
        group_state: Any,
        local_column_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.quantize_column(column, group_state, local_column_index), None

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        raise NotImplementedError
