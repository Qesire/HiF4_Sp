from __future__ import annotations

from dataclasses import dataclass, field

from ...algorithms.gptq_original import OriginalGPTQConfig


@dataclass(frozen=True)
class HiF4Stage1Config:
    enabled: bool = True
    scale_neighbors: int = 8

    def validate(self) -> None:
        if self.scale_neighbors < 0:
            raise ValueError("stage1 scale_neighbors cannot be negative")


@dataclass(frozen=True)
class HiF4Stage2Config:
    enabled: bool = True
    num_sweeps: int = 1
    projection_neighbors: int = 2
    use_deviation_correlation: bool = False
    group_order: str = "forward"

    def validate(self) -> None:
        if self.num_sweeps <= 0:
            raise ValueError("stage2 num_sweeps must be positive")
        if self.projection_neighbors < 0:
            raise ValueError("projection_neighbors cannot be negative")
        if self.group_order not in {"forward", "reverse", "alternating"}:
            raise ValueError("unsupported stage2 group_order")


@dataclass(frozen=True)
class TwoStageHiF4GPTQConfig:
    original_gptq: OriginalGPTQConfig = field(default_factory=OriginalGPTQConfig)
    stage1: HiF4Stage1Config = field(default_factory=HiF4Stage1Config)
    stage2: HiF4Stage2Config = field(default_factory=HiF4Stage2Config)

    def validate(self) -> None:
        self.original_gptq.validate()
        self.stage1.validate()
        self.stage2.validate()
