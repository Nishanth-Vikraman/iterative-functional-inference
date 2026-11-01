import torch

from ifi.config import (
    ExperimentConfig,
    OperatorConfig,
    PhaseAConfig,
    PhaseBConfig,
    PredictiveStateConfig,
    TaskConfig,
)
from ifi.operators import build_function_system
from ifi.tasks import SyntheticFunctionBenchmark
from ifi.v22_models import FixedResidualContrast, PredictiveRecurrentPolicy
from ifi.v22_phase_b import PredictiveStateTrainer


def _config():
    return ExperimentConfig(
        device="cpu",
        require_cuda=False,
        tasks=TaskConfig(support_sizes=(4,)),
        operator=OperatorConfig(
            dimension=12, action_dimension=4, hidden_dimension=16, basis_depth=1
        ),
        phase_a=PhaseAConfig(span_threshold=100.0),
        phase_b=PhaseBConfig(
            batch_size=3, rollout_max=2, variable_rollout=False,
            probe_size=12, latent_dimension=12, controller_hidden_dimension=16,
        ),
        predictive_state=PredictiveStateConfig(temperature=0.2),
    )


def test_predictive_policy_has_disjoint_t_u_and_no_semantic_residual_encoder():
    policy = PredictiveRecurrentPolicy(12, 4, 4)
    t_ids = {id(parameter) for parameter in policy.t_parameters()}
    u_ids = {id(parameter) for parameter in policy.u_parameters()}
    assert t_ids
    assert u_ids
    assert t_ids.isdisjoint(u_ids)
    assert not hasattr(policy, "residual_encoder")
    assert policy.action_controller.raw_residual_projection.in_features == 6
    assert policy.state_updater.raw_residual_projection.in_features == 6


def test_fixed_residual_contrast_has_no_parameters_and_is_deterministic():
    first = FixedResidualContrast(12, seed=2202, temperature=0.1)
    second = FixedResidualContrast(12, seed=2202, temperature=0.1)
    assert list(first.parameters()) == []
    torch.testing.assert_close(first.state_projection, second.state_projection)
    tokens = torch.randn(4, 6)
    assert first.residual_signature(tokens).shape == (24,)


def test_gradient_routing_is_exactly_separated_and_operator_stays_frozen():
    torch.manual_seed(8)
    config = _config()
    system, operator = build_function_system(config.operator, torch.device("cpu"))
    operator.freeze()
    policy = PredictiveRecurrentPolicy(12, 4, 4)
    benchmark = SyntheticFunctionBenchmark(config.tasks, 8)
    trainer = PredictiveStateTrainer(config, system, operator, benchmark, policy)
    metrics = trainer.train_step(diagnose_routing=True)
    assert metrics["teacher_to_T_grad_norm"] > 0
    assert metrics["teacher_to_U_grad_norm"] == 0
    assert metrics["info_to_T_grad_norm"] == 0
    assert metrics["info_to_U_grad_norm"] > 0
    assert metrics["infonce_loss"] > 0
    assert all(parameter.grad is None for parameter in operator.parameters())


def test_state_update_is_not_forced_to_fixed_layernorm_radius():
    torch.manual_seed(9)
    policy = PredictiveRecurrentPolicy(12, 4, 4)
    state = torch.zeros(12)
    tokens = torch.randn(4, 6)
    action = torch.randn(4)
    first = policy.update_state(state, tokens, action)
    second = policy.update_state(first.detach(), tokens * 2, action)
    assert not torch.allclose(first.norm(), torch.tensor(12.0).sqrt())
    assert not torch.allclose(first.norm(), second.norm())
