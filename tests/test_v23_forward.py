import torch

from ifi.config import (
    ExperimentConfig,
    ForwardStateConfig,
    OperatorConfig,
    PhaseAConfig,
    PhaseBConfig,
    TaskConfig,
)
from ifi.operators import build_function_system
from ifi.tasks import SyntheticFunctionBenchmark
from ifi.v23_models import ForwardTrainedPolicy
from ifi.v23_phase_b import ForwardCreditTrainer


def _config() -> ExperimentConfig:
    return ExperimentConfig(
        device="cpu",
        require_cuda=False,
        tasks=TaskConfig(support_sizes=(4,)),
        operator=OperatorConfig(
            dimension=12,
            action_dimension=4,
            hidden_dimension=16,
            basis_depth=1,
        ),
        phase_a=PhaseAConfig(span_threshold=100.0),
        phase_b=PhaseBConfig(
            batch_size=3,
            rollout_max=3,
            variable_rollout=False,
            probe_size=12,
            latent_dimension=12,
            controller_hidden_dimension=16,
        ),
        forward_state=ForwardStateConfig(block_length=1, extended_rollout_steps=4),
    )


def _trainer() -> tuple[ForwardCreditTrainer, ForwardTrainedPolicy]:
    config = _config()
    system, operator = build_function_system(config.operator, torch.device("cpu"))
    operator.freeze()
    policy = ForwardTrainedPolicy(12, 4, 4)
    benchmark = SyntheticFunctionBenchmark(config.tasks, 23)
    return ForwardCreditTrainer(config, system, operator, benchmark, policy), policy


def _snapshot(parameters: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in parameters]


def _changed(before: list[torch.Tensor], parameters: list[torch.nn.Parameter]) -> bool:
    return any(not torch.equal(old, new.detach()) for old, new in zip(before, parameters))


def test_forward_policy_is_one_controller_without_er_u_or_fixed_state_norm():
    torch.manual_seed(31)
    policy = ForwardTrainedPolicy(12, 4, 4)
    action_ids = {id(parameter) for parameter in policy.action_parameters()}
    h_ids = {id(parameter) for parameter in policy.h_parameters()}
    assert action_ids
    assert h_ids
    assert action_ids.isdisjoint(h_ids)
    assert not hasattr(policy, "residual_encoder")
    assert not hasattr(policy, "state_updater")
    assert policy.controller.raw_residual_projection.in_features == 6

    raw = torch.randn(4, 6)
    z_data = torch.randn(12)
    state = torch.zeros(12)
    first = policy.step(raw, z_data, state, ()).next_state
    second = policy.step(raw * 1.7, z_data, first.detach(), ()).next_state
    assert not torch.allclose(first.norm(), torch.tensor(12.0).sqrt())
    assert not torch.allclose(first.norm(), second.norm())


def test_h_step_updates_only_state_head_and_has_hard_environment_boundary():
    torch.manual_seed(32)
    trainer, policy = _trainer()
    h_before = _snapshot(policy.h_parameters())
    action_before = _snapshot(policy.action_parameters())

    metrics = trainer.train_step(1)

    assert metrics["block_h"] == 1.0
    assert metrics["H_h_grad_norm"] > 0.0
    assert metrics["H_frozen_action_grad_norm"] == 0.0
    assert metrics["H_phi_i_grad_norm"] == 0.0
    assert metrics["H_environment_grad_path"] == 0.0
    assert _changed(h_before, policy.h_parameters())
    assert not _changed(action_before, policy.action_parameters())
    assert all(parameter.grad is None for parameter in trainer.operator.parameters())


def test_phi_step_updates_only_action_path_with_detached_recurrent_inputs():
    torch.manual_seed(33)
    trainer, policy = _trainer()
    trainer.train_step(1)
    h_before = _snapshot(policy.h_parameters())
    action_before = _snapshot(policy.action_parameters())

    metrics = trainer.train_step(2)

    assert metrics["block_phi"] == 1.0
    assert metrics["Phi_action_grad_norm"] > 0.0
    assert metrics["Phi_h_grad_norm"] == 0.0
    assert metrics["Phi_state_inputs_require_grad"] == 0.0
    assert not _changed(h_before, policy.h_parameters())
    assert _changed(action_before, policy.action_parameters())
    assert all(parameter.grad is None for parameter in trainer.operator.parameters())


def test_coordinate_block_schedule_is_frozen_and_contiguous():
    trainer, _ = _trainer()
    trainer.config.forward_state.block_length = 2
    assert [trainer.block_for_step(step) for step in range(1, 9)] == [
        "h",
        "h",
        "phi",
        "phi",
        "h",
        "h",
        "phi",
        "phi",
    ]
