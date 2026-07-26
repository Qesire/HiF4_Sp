from __future__ import annotations

import torch

from ..state import CurvatureStats


class CurvatureCollector:
    """通用 X^T X 累加器，不绑定 GPTQ 或 MagR。"""

    def __init__(
        self,
        in_features: int,
        *,
        device: str | torch.device = "cpu",
        sample_tokens: int | None = None,
        max_tokens: int | None = None,
        module_name: str | None = None,
        random_seed: int = 0,
    ) -> None:
        self.in_features = int(in_features)
        self.device = torch.device(device)
        self.sample_tokens = None if sample_tokens is None else int(sample_tokens)
        self.max_tokens = None if max_tokens is None else int(max_tokens)
        self.module_name = module_name
        self.random_seed = int(random_seed)
        self.h_sum = torch.zeros(
            self.in_features,
            self.in_features,
            dtype=torch.float32,
            device=self.device,
        )
        self.nsamples = 0
        self.num_calls = 0

    @torch.no_grad()
    def add_batch(self, x: torch.Tensor) -> None:
        if x.numel() == 0:
            return
        if x.shape[-1] != self.in_features:
            raise ValueError(f"input shape={tuple(x.shape)}, expected last dim={self.in_features}")

        x2 = x.detach().reshape(-1, self.in_features)
        if self.sample_tokens and x2.shape[0] > self.sample_tokens:
            generator = torch.Generator(device=x2.device)
            generator.manual_seed(self.random_seed + self.num_calls)
            indices = torch.randperm(x2.shape[0], generator=generator, device=x2.device)[
                : self.sample_tokens
            ]
            x2 = x2.index_select(0, indices)
        elif self.max_tokens and x2.shape[0] > self.max_tokens:
            x2 = x2[: self.max_tokens]

        x2 = x2.to(device=self.device, dtype=torch.float32)
        self.h_sum.add_(x2.T @ x2)
        self.nsamples += int(x2.shape[0])
        self.num_calls += 1

    def export(self, *, normalize: bool = True) -> CurvatureStats:
        if self.nsamples <= 0:
            raise RuntimeError("curvature collector has no samples")
        matrix = self.h_sum / float(self.nsamples) if normalize else self.h_sum.clone()
        return CurvatureStats(
            matrix=matrix.detach().cpu().contiguous(),
            nsamples=self.nsamples,
            in_features=self.in_features,
            normalization="mean_xtx" if normalize else "sum_xtx",
            domain="original",
            module_name=self.module_name,
        )
