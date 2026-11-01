from __future__ import annotations

from pathlib import Path

import torch

from .config import ExperimentConfig
from .models import IFIPolicy, StaticHypernetwork


def save_phase_b(
    path: str | Path,
    config: ExperimentConfig,
    ifi: IFIPolicy,
    ar: IFIPolicy,
    static: StaticHypernetwork,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": config.to_dict(),
        "ifi": ifi.state_dict(),
        "ar": ar.state_dict(),
        "static": static.state_dict(),
    }, path)


def load_policy_states(
    path: str | Path,
    ifi: IFIPolicy,
    ar: IFIPolicy,
    static: StaticHypernetwork,
    device: torch.device,
) -> None:
    payload = torch.load(path, map_location=device, weights_only=False)
    ifi.load_state_dict(payload["ifi"])
    ar.load_state_dict(payload["ar"])
    static.load_state_dict(payload["static"])

