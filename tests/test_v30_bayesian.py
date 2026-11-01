import torch

from ifi.functions import ComposedFunction
from ifi.operators import build_function_system
from ifi.teachers import BayesianLatentSystemTeacher, Teacher
from ifi.v30_config import V30Config, V30PhaseBConfig, V30TaskConfig
from ifi.v30_models import HistoryItem, V30Policy, parameter_counts
from ifi.v30_phase_b import V30PhaseBTrainer
from ifi.v30_tasks import (
    BayesianLatentBenchmark,
    build_latent_library,
    latent_split_indices,
)


def _config() -> V30Config:
    return V30Config(
        device="cpu",
        require_cuda=False,
        tasks=V30TaskConfig(
            support_sizes=(4,),
            train_sequence_length=3,
            evaluation_sequence_length=4,
        ),
        phase_b=V30PhaseBConfig(
            steps=1,
            batch_size=2,
            rollout_max=3,
            variable_rollout=False,
            probe_size=12,
            model_dimension=16,
        ),
    )


def _policies(config: V30Config):
    return {
        name: V30Policy(name, 16, config.operator.action_dimension, 4, 4)
        for name in (
            "direct_current", "full_history", "query_history", "ar_no_feedback"
        )
    }


def test_latent_split_withholds_joint_combinations_but_preserves_marginals():
    config = V30TaskConfig()
    library = build_latent_library(config, torch.device("cpu"))
    splits = latent_split_indices(library)
    assert library.size == 32
    assert set(splits["train_latents"]).isdisjoint(splits["heldout_latents"])
    assert len(splits["train_latents"]) + len(splits["heldout_latents"]) == 32
    for column in range(4):
        all_values = set(library.coefficients[:, column].tolist())
        for split in splits.values():
            values = set(library.coefficients[list(split), column].tolist())
            assert values == all_values


def test_bayesian_teacher_is_generic_exact_and_does_not_need_latent_identity():
    config = V30TaskConfig(support_sizes=(4,), train_sequence_length=2)
    benchmark = BayesianLatentBenchmark(config, 9, "cpu")
    episode = benchmark.sample_episode(4, "train_latents", 2)
    teacher = BayesianLatentSystemTeacher(
        benchmark.library, config.likelihood_noise
    )
    assert isinstance(teacher, Teacher)
    posterior = teacher.posterior(episode.cumulative(1))
    torch.testing.assert_close(posterior.sum(), torch.tensor(1.0))
    system, operator = build_function_system(V30Config().operator, torch.device("cpu"))
    current = ComposedFunction(system, operator)
    probes = torch.linspace(-2, 2, 16).unsqueeze(-1)
    update = teacher.propose_update(current, episode.cumulative(1), probes)
    torch.testing.assert_close(
        update.teacher(probes), teacher.predictive(posterior, probes)
    )
    assert "posterior_entropy" in update.metadata
    assert "latent_index" not in update.metadata


def test_explicit_history_models_have_no_er_recurrence_or_state_updater():
    config = _config()
    policies = _policies(config)
    counts = parameter_counts(policies)
    assert counts["direct_current"] == counts["full_history"]
    relative_gap = abs(counts["query_history"] - counts["full_history"]) / counts["full_history"]
    assert relative_gap < 0.20
    for policy in policies.values():
        assert not hasattr(policy, "residual_encoder")
        assert not hasattr(policy, "state_updater")
        assert not hasattr(policy, "recurrent_state")


def test_history_changes_a_and_b_but_not_direct_current():
    torch.manual_seed(12)
    policies = _policies(_config())
    raw = torch.randn(4, 6)
    old_raw = torch.randn(4, 6)
    action = torch.randn(12)
    history = (HistoryItem(old_raw, action),)
    for name in ("direct_current", "full_history", "query_history"):
        policy = policies[name]
        z_data = policy.encode_data(torch.randn(4, 1), torch.randn(4, 1))
        without = policy.predict(raw, z_data, ())
        with_history = policy.predict(raw, z_data, history)
        if name == "direct_current":
            torch.testing.assert_close(without, with_history)
        else:
            assert not torch.allclose(without, with_history)


def test_phase_b_local_training_keeps_operator_frozen():
    torch.manual_seed(13)
    config = _config()
    system, operator = build_function_system(config.operator, torch.device("cpu"))
    operator.freeze()
    benchmark = BayesianLatentBenchmark(config.tasks, 13, "cpu")
    trainer = V30PhaseBTrainer(
        config, system, operator, benchmark, _policies(config)
    )
    metrics = trainer.train_step()
    assert metrics["operator_gradient"] == 0.0
    assert metrics["gradient_norm"] > 0.0
    assert all(parameter.grad is None for parameter in operator.parameters())
