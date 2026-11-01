from __future__ import annotations

import torch

from ifi.v30_config import V30Config
from ifi.v31_config import V31Config
from ifi.v31_targets import (
    V31OracleTeacher,
    build_target_registry,
    reconstruct_training_support,
    registry_groups,
)


def _registry():
    config = V30Config.from_json("runs/v3_0_bayesian/config.resolved.json")
    support = reconstruct_training_support(config, torch.device("cpu"))
    return config, support, build_target_registry(support)


def test_registry_is_deterministic_complete_and_finite():
    _, _, targets = _registry()
    assert len(targets) == 45
    assert len({target.target_id for target in targets}) == len(targets)
    assert {target.bucket for target in targets} == {
        "heldout_combinations",
        "coefficient_extrapolation",
        "structural_shift",
        "compositional_shift",
    }
    x = torch.linspace(-2, 2, 257).unsqueeze(-1)
    assert all(torch.isfinite(target.evaluate(x)).all() for target in targets)


def test_coefficient_targets_are_outside_persisted_support():
    _, support, targets = _registry()
    coefficient_support = support["coefficient_support"]
    for target in targets:
        if target.bucket != "coefficient_extrapolation":
            continue
        outside = 0
        for key, value in target.parameters.items():
            limits = coefficient_support[key]
            outside += int(value < limits["min"] or value > limits["max"])
        assert outside == target.factor_count


def test_oracle_is_exact_distribution_over_locked_group():
    config, _, targets = _registry()
    group = registry_groups(targets)[("compositional_shift", "mild")]
    oracle = V31OracleTeacher(group, config.tasks.likelihood_noise)
    x = torch.tensor([[-1.0], [0.5]])
    y = group[0].evaluate(x)
    posterior = oracle.posterior((x, y))
    assert posterior.shape == (len(group),)
    assert torch.allclose(posterior.sum(), torch.tensor(1.0))
    assert torch.isfinite(oracle.predictive(posterior, x)).all()


def test_rollout_ladder_is_three_times_training_depth():
    train = V30Config.from_json("runs/v3_0_bayesian/config.resolved.json")
    config = V31Config.from_json("configs/v3_1_ood.json")
    config.validate(train.phase_b.rollout_max, train.evaluation.extended_rollout_steps)
    assert config.rollout_steps == 3 * train.phase_b.rollout_max
