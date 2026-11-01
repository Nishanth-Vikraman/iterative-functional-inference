from __future__ import annotations

import torch

from ifi.v32_losses import (
    LowRankFunctionalGeometry,
    normalized_functional_loss,
    normalized_mahalanobis_loss,
    posterior_function_perturbations,
    teacher_geometry,
)


def test_normalized_functional_zero_and_scale_invariance():
    target = torch.tensor([1.0, -2.0, 0.5])
    zero = normalized_functional_loss(target, target, 1e-12)[0]
    first = normalized_functional_loss(1.2 * target, target, 1e-12)[0]
    second = normalized_functional_loss(3.6 * target, 3.0 * target, 1e-12)[0]
    assert float(zero) == 0.0
    assert torch.allclose(first, second)


def test_mahalanobis_penalizes_low_variance_direction():
    geometry = LowRankFunctionalGeometry(
        torch.tensor([10.0, 0.1]), torch.eye(2), 0.01, 2, 2.0, 100.0, 1001.0
    )
    high_variance = geometry.quadratic(torch.tensor([1.0, 0.0]))
    low_variance = geometry.quadratic(torch.tensor([0.0, 1.0]))
    assert low_variance > high_variance
    zero = normalized_mahalanobis_loss(
        torch.ones(2), torch.ones(2), geometry, 1e-12
    )[0]
    assert float(zero) == 0.0


def test_covariance_geometry_is_finite_psd_and_regularized():
    perturbations = torch.tensor([
        [1.0, 0.0, 0.5],
        [-1.0, 0.0, -0.5],
        [0.5, 0.2, 0.0],
        [-0.5, -0.2, 0.0],
    ])
    covariance = torch.cov(perturbations.T)
    assert torch.linalg.eigvalsh(covariance).min() >= -1e-6
    geometry = teacher_geometry(perturbations, 1e-3, 1e-7)
    assert geometry.raw_rank > 0
    assert geometry.condition_after > 0
    assert torch.isfinite(geometry.quadratic(torch.ones(3)))


def test_logit_perturbations_stay_in_valid_bayesian_function_space():
    posterior = torch.tensor([0.2, 0.3, 0.5])
    hypotheses = torch.tensor([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]])
    noise = torch.tensor([[0.1, -0.1, 0.0], [-0.1, 0.1, 0.0]])
    changes = posterior_function_perturbations(posterior, hypotheses, noise)
    assert changes.shape == (2, 2)
    assert torch.isfinite(changes).all()


def test_functional_loss_routes_gradient_to_action_not_frozen_operator():
    from ifi.operators import build_function_system
    from ifi.v30_config import V30Config

    config = V30Config.from_json("runs/v3_0_bayesian/config.resolved.json")
    system, operator = build_function_system(config.operator, torch.device("cpu"))
    checkpoint = torch.load("runs/v3_0_bayesian/phase_a.pt", map_location="cpu", weights_only=False)
    operator.load_state_dict(checkpoint["operator"])
    operator.freeze()
    x = torch.linspace(-2, 2, 32).unsqueeze(-1)
    action = torch.randn(config.operator.action_dimension) * 0.01
    action.requires_grad_()
    latent = system.encode(x)
    prediction = system.decode(operator(latent, action))
    target = torch.sin(x)
    loss = normalized_functional_loss(prediction, target, 1e-8)[0]
    loss.backward()
    assert action.grad is not None and float(action.grad.norm()) > 0
    assert all(parameter.grad is None for parameter in operator.parameters())
