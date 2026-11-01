import torch

from ifi.config import (
    EvalConfig,
    ExperimentConfig,
    OperatorConfig,
    PhaseAConfig,
    PhaseBConfig,
    TaskConfig,
)
from ifi.functions import ComposedFunction
from ifi.inference import adapt_without_gradients
from ifi.models import IFIPolicy, StaticHypernetwork
from ifi.operators import build_function_system
from ifi.phase_b import PhaseBTrainer
from ifi.tasks import SyntheticFunctionBenchmark


def tiny_config():
    return ExperimentConfig(
        device="cpu",
        require_cuda=False,
        tasks=TaskConfig(support_sizes=(4,)),
        operator=OperatorConfig(
            dimension=12, action_dimension=4, hidden_dimension=16, basis_depth=1
        ),
        phase_a=PhaseAConfig(span_threshold=100.0),
        phase_b=PhaseBConfig(
            batch_size=1, rollout_max=2, probe_size=16, latent_dimension=12,
            controller_hidden_dimension=16,
        ),
        evaluation=EvalConfig(episodes_per_cell=1, seeds=(1,)),
    )


def make_models(config):
    ifi = IFIPolicy(12, 4, 16, True)
    ar = IFIPolicy(12, 4, 16, False)
    static = StaticHypernetwork(12, 4, 2)
    return ifi, ar, static


def test_phase_b_never_updates_frozen_operator():
    torch.manual_seed(2)
    config = tiny_config()
    system, operator = build_function_system(config.operator, torch.device("cpu"))
    operator.freeze()
    before = {name: value.clone() for name, value in operator.state_dict().items()}
    benchmark = SyntheticFunctionBenchmark(config.tasks, 2)
    ifi, ar, static = make_models(config)
    trainer = PhaseBTrainer(config, system, operator, benchmark, ifi, ar, static)
    metrics = trainer.train_step()
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    for name, value in operator.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in operator.parameters())


def test_teacher_free_inference_has_no_gradients_and_recomputes_feedback():
    config = tiny_config()
    system, operator = build_function_system(config.operator, torch.device("cpu"))
    operator.freeze()
    ifi, _, _ = make_models(config)
    benchmark = SyntheticFunctionBenchmark(config.tasks, 4)
    episode = benchmark.sample_episode(4)
    trace = adapt_without_gradients(
        ifi, ComposedFunction(system, operator), episode.support_x, episode.support_y, 3
    )
    assert len(trace.functions) == 4
    assert len(trace.actions) == 3
    assert all(not action.requires_grad for action in trace.actions)
    assert all(function.depth == index for index, function in enumerate(trace.functions))


def test_ar_policy_ignores_fresh_residual_feedback():
    ar = IFIPolicy(12, 4, 16, False)
    x = torch.linspace(-1, 1, 4).unsqueeze(-1)
    y_a = torch.zeros_like(x)
    y_b = torch.ones_like(x) * 10
    prediction = torch.zeros_like(x)
    torch.testing.assert_close(
        ar.encode_residual(x, y_a, prediction), ar.encode_residual(x, y_b, prediction)
    )

