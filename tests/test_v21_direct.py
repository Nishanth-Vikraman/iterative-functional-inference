import torch

from ifi.config import ExperimentConfig, OperatorConfig, PhaseAConfig, PhaseBConfig, TaskConfig
from ifi.functions import ComposedFunction
from ifi.operators import build_function_system
from ifi.tasks import SyntheticFunctionBenchmark
from ifi.v21_evaluation import direct_adapt
from ifi.v21_models import DirectIFIPolicy, trainable_parameter_count
from ifi.v21_phase_b import DirectPhaseBTrainer


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
            batch_size=1, rollout_max=2, probe_size=12,
            latent_dimension=12, controller_hidden_dimension=16,
        ),
    )


def _policies():
    return {
        name: DirectIFIPolicy(name, 12, 4, 4)
        for name in ("direct_current", "direct_history", "direct_recurrent")
    }


def test_direct_variants_have_exactly_matched_counts_and_no_residual_encoder():
    policies = _policies()
    assert len({trainable_parameter_count(policy) for policy in policies.values()}) == 1
    assert all(not hasattr(policy, "residual_encoder") for policy in policies.values())
    assert all(policy.controller.raw_residual_projection.in_features == 6 for policy in policies.values())


def test_ablation_inputs_obey_preregistered_contract():
    torch.manual_seed(4)
    policies = _policies()
    tokens = torch.randn(4, 6)
    z_data = torch.randn(12)
    history = (torch.randn(4),)
    state = torch.randn(12)

    a_with = policies["direct_current"].step(tokens, z_data, history, state).action
    a_without = policies["direct_current"].step(tokens, z_data, (), None).action
    torch.testing.assert_close(a_with, a_without)

    b_with = policies["direct_history"].step(tokens, z_data, history, state).action
    b_without = policies["direct_history"].step(tokens, z_data, (), None).action
    assert not torch.allclose(b_with, b_without)
    b_other_state = policies["direct_history"].step(tokens, z_data, history, -state).action
    torch.testing.assert_close(b_with, b_other_state)

    c_with = policies["direct_recurrent"].step(tokens, z_data, history, state)
    c_other = policies["direct_recurrent"].step(tokens, z_data, history, -state)
    assert c_with.next_state is not None
    assert not torch.allclose(c_with.action, c_other.action)


def test_direct_phase_b_keeps_operator_frozen_and_inference_is_gradient_free():
    torch.manual_seed(5)
    config = _config()
    system, operator = build_function_system(config.operator, torch.device("cpu"))
    operator.freeze()
    policies = _policies()
    benchmark = SyntheticFunctionBenchmark(config.tasks, 5)
    trainer = DirectPhaseBTrainer(config, system, operator, benchmark, policies)
    metrics = trainer.train_step()
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert all(parameter.grad is None for parameter in operator.parameters())

    episode = benchmark.sample_episode(4)
    trace = direct_adapt(
        policies["direct_recurrent"],
        ComposedFunction(system, operator),
        episode.support_x,
        episode.support_y,
        3,
    )
    assert len(trace.functions) == 4
    assert all(not action.requires_grad for action in trace.actions)
    assert all(tokens.shape == (4, 6) for tokens in trace.residual_tokens)
    assert all(state is not None for state in trace.states_after)

