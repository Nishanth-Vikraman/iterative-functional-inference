import json

import pytest
import torch

from ifi.config import ExperimentConfig, TaskConfig
from ifi.tasks import SyntheticFunctionBenchmark


def test_novelty_splits_are_explicit_and_finite():
    benchmark = SyntheticFunctionBenchmark(TaskConfig(), seed=9)
    families = {}
    for split in benchmark.SPLITS:
        episode = benchmark.sample_episode(8, split)
        families[split] = episode.target.family
        assert episode.support_x.shape == (8, 1)
        assert torch.isfinite(episode.support_y).all()
    assert families["heldout_combinations"] == "heldout_mixture"
    assert families["heldout_compositions"] == "composition"


def test_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"phase_b": {"not_a_setting": 1}}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown PhaseBConfig keys"):
        ExperimentConfig.from_json(path)

