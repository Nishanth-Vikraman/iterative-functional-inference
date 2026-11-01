from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from .v30_config import V30TaskConfig


LatentSplit = Literal["train_latents", "heldout_latents"]


@dataclass(frozen=True)
class LatentLibrary:
    coefficients: Tensor

    def evaluate(self, x: Tensor) -> Tensor:
        """Return [hypothesis, point, 1] values for every finite latent hypothesis."""
        coefficients = self.coefficients.to(x.device)
        a, b, c, omega = (coefficients[:, index] for index in range(4))
        return (
            a[:, None, None] * x[None]
            + b[:, None, None] * x[None].square()
            + c[:, None, None] * torch.sin(omega[:, None, None] * x[None])
        )

    def function(self, latent_index: int, x: Tensor) -> Tensor:
        return self.evaluate(x)[latent_index]

    @property
    def size(self) -> int:
        return self.coefficients.shape[0]


@dataclass(frozen=True)
class EvidenceBatch:
    x: Tensor
    y: Tensor
    sensor_index: int


@dataclass(frozen=True)
class BayesianEpisode:
    latent_index: int
    split: LatentSplit
    batches: tuple[EvidenceBatch, ...]
    episode_id: int

    def cumulative(self, iteration: int) -> tuple[Tensor, Tensor]:
        selected = self.batches[: iteration + 1]
        return (
            torch.cat([batch.x for batch in selected], dim=0),
            torch.cat([batch.y for batch in selected], dim=0),
        )


def build_latent_library(config: V30TaskConfig, device: torch.device) -> LatentLibrary:
    values = list(itertools.product(
        config.coefficients_a,
        config.coefficients_b,
        config.coefficients_c,
        config.frequencies,
    ))
    return LatentLibrary(torch.tensor(values, dtype=torch.float32, device=device))


def latent_split_indices(library: LatentLibrary) -> dict[LatentSplit, tuple[int, ...]]:
    # Checkerboard holdout preserves marginals while withholding joint configurations.
    train = []
    heldout = []
    for index, coefficients in enumerate(library.coefficients.cpu().tolist()):
        value_indices = [
            sorted(set(library.coefficients[:, column].cpu().tolist())).index(value)
            for column, value in enumerate(coefficients)
        ]
        target = heldout if sum(value_indices) % 4 == 0 else train
        target.append(index)
    return {"train_latents": tuple(train), "heldout_latents": tuple(heldout)}


class BayesianLatentBenchmark:
    def __init__(
        self,
        config: V30TaskConfig,
        seed: int,
        device: str | torch.device,
    ):
        self.config = config
        self.device = torch.device(device)
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.library = build_latent_library(config, self.device)
        self.split_indices = latent_split_indices(self.library)
        self._next_episode = 0

    def _sensor_points(self, sensor_index: int, support_size: int) -> Tensor:
        center = self.config.sensor_centers[sensor_index]
        offsets = torch.linspace(
            -self.config.sensor_width,
            self.config.sensor_width,
            support_size,
        ) if support_size > 1 else torch.zeros(1)
        return (center + offsets).unsqueeze(-1).to(self.device)

    def sample_episode(
        self,
        support_size: int,
        split: LatentSplit,
        sequence_length: int,
    ) -> BayesianEpisode:
        indices = self.split_indices[split]
        latent_index = indices[
            int(torch.randint(len(indices), (), generator=self.generator).item())
        ]
        batches = []
        for _ in range(sequence_length):
            sensor_index = int(torch.randint(
                len(self.config.sensor_centers), (), generator=self.generator
            ).item())
            x = self._sensor_points(sensor_index, support_size)
            y = self.library.function(latent_index, x)
            noise = torch.randn(y.shape, generator=self.generator).to(self.device)
            y = y + self.config.observation_noise * noise
            batches.append(EvidenceBatch(x, y, sensor_index))
        episode = BayesianEpisode(
            latent_index, split, tuple(batches), self._next_episode
        )
        self._next_episode += 1
        return episode

    def sample_query(self, size: int) -> Tensor:
        x = torch.rand((size, 1), generator=self.generator)
        x = self.config.x_min + (self.config.x_max - self.config.x_min) * x
        return x.to(self.device)

    def posterior(self, support: tuple[Tensor, Tensor]) -> Tensor:
        x, y = support
        predictions = self.library.evaluate(x).squeeze(-1)
        residual = y.squeeze(-1).unsqueeze(0) - predictions
        log_likelihood = -0.5 * residual.square().sum(dim=-1) / (
            self.config.likelihood_noise ** 2
        )
        return torch.softmax(log_likelihood, dim=0)

    def posterior_predictive(self, posterior: Tensor, x: Tensor) -> Tensor:
        return torch.einsum("h,hnd->nd", posterior, self.library.evaluate(x))
