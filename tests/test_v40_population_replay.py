from dataclasses import replace
from pathlib import Path

import torch

from ifi.v40_config import V40Config
from ifi.v40_models import PopulationController, normalized_residual
from ifi.v40_population import ReplayPopulation
from ifi.v40_training import PopulationTrainer, build_models, integrity_audit
from ifi.v40_worlds import (
    analytic_action,
    sample_well_conditioned_worlds,
    sample_probe_actions,
    transition,
)
from ifi.v41_history_policy_eval import HistoryPolicy, append_with_policy
from ifi.v40_evaluation import make_rollout_state
from ifi.v43_config import V43Config
from ifi.v43_convergence_replay import (
    MixedReplay,
    PostTrainer,
    archive_history_error,
    harvest_convergence_pool,
)
from ifi.v44_config import V44Config
from ifi.v44_populations import RecoveryPopulation
from ifi.v44_training import ArmRuntime, arm_config, parameter_count
from ifi.v45_config import V45Config
from ifi.v45_replay import collect_current_policy, mixed_replay_batch


def tiny_config() -> V40Config:
    return replace(
        V40Config(device="cpu", require_cuda=False),
        pool_size=64,
        batch_size=8,
        history_capacity=6,
        max_age=100,
    )


def test_world_family_is_bounded_and_oracle_solves() -> None:
    generator = torch.Generator().manual_seed(4)
    worlds = sample_well_conditioned_worlds(32, 8, 0.5, 2.0, generator, 100)
    singular = torch.linalg.svdvals(worlds.matrices)
    assert float(singular.min()) >= 0.5 - 1e-5
    assert float(singular.max()) <= 2.0 + 1e-5
    assert float((singular.max(-1).values / singular.min(-1).values).max()) <= 4.0 + 1e-4
    residual = torch.randn(32, 8)
    solved = transition(residual, worlds.matrices, analytic_action(worlds.matrices, residual))
    assert float(solved.abs().max()) < 2e-6


def test_scale_free_residual_representation() -> None:
    residual = torch.tensor([[3.0, 4.0]])
    direction, magnitude = normalized_residual(residual, 1.0, 1e-8)
    scaled_direction, scaled_magnitude = normalized_residual(10 * residual, 1.0, 1e-8)
    assert torch.allclose(direction, scaled_direction, atol=1e-7)
    assert torch.all((magnitude > 0) & (magnitude < 1))
    assert torch.all((scaled_magnitude > 0) & (scaled_magnitude < 1))


def test_eight_bootstrap_probes_are_orthogonal_and_full_rank() -> None:
    generator = torch.Generator().manual_seed(8)
    probes = sample_probe_actions(16, 8, 8, 0.35, generator, orthogonal=True)
    gram = probes @ probes.transpose(-1, -2)
    target = 0.35**2 * torch.eye(8).expand(16, -1, -1)
    assert torch.allclose(gram, target, atol=2e-6, rtol=2e-6)
    assert torch.all(torch.linalg.matrix_rank(probes) == 8)


def test_population_histories_are_real_and_detached() -> None:
    config = tiny_config()
    population = ReplayPopulation(config, "query_history", 9)
    assert population.actual_history_max_error() < 2e-6
    indices = population.sample_indices(config.batch_size)
    batch = population.controller_batch(indices, torch.device("cpu"))
    assert not batch.current.requires_grad
    assert not batch.history.requires_grad
    action = torch.zeros(config.batch_size, config.action_dimension)
    next_residual = transition(batch.current, population.matrix_batch(indices, torch.device("cpu")), action)
    population.apply(indices, action, next_residual, 1)
    assert population.actual_history_max_error() < 2e-6


def test_history_attention_is_permutation_invariant_and_has_no_clock() -> None:
    config = tiny_config()
    models = build_models(config, torch.device("cpu"))
    populations = {
        name: ReplayPopulation(config, name, 10) for name in models
    }
    audit = integrity_audit(config, models, populations, torch.device("cpu"))
    assert audit["passed"]
    assert audit["checks"]["permutation_invariant_history"]
    assert audit["checks"]["no_positional_parameters"]


def test_local_loss_reaches_policy_but_not_stored_state() -> None:
    config = tiny_config()
    population = ReplayPopulation(config, "query_history", 11)
    model = PopulationController("query_history", config)
    indices = population.sample_indices(config.batch_size)
    batch = population.controller_batch(indices, torch.device("cpu"))
    matrix = population.matrix_batch(indices, torch.device("cpu"))
    action = model(batch)
    next_residual = transition(batch.current, matrix, action)
    loss = (
        next_residual.square().sum(-1)
        / (batch.current.square().sum(-1) + config.numerical_epsilon)
    ).mean()
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert not batch.current.requires_grad
    assert not batch.history.requires_grad
    assert not matrix.requires_grad


def test_asynchronous_sampling_creates_mixed_real_ages() -> None:
    config = tiny_config()
    population = ReplayPopulation(config, "query_history", 12)
    for step in range(1, 25):
        indices = population.sample_indices(config.batch_size)
        before = population.current[indices]
        action = torch.zeros(config.batch_size, config.action_dimension)
        after = transition(before, population.matrices[indices], action)
        population.apply(indices, action, after, step)
    assert population.ages.unique().numel() > 3
    assert int(population.ages.max()) > int(population.ages.min())


def test_margin_objective_runs_with_finite_unclipped_gradient() -> None:
    config = replace(
        tiny_config(),
        history_capacity=16,
        bootstrap_actions=8,
        bootstrap_orthogonal=True,
        loss_kind="softplus_progress",
        progress_tau=0.95,
        action_regularization=0.0,
        gradient_clip=0.0,
        learning_rate=1e-4,
    )
    trainer = PopulationTrainer(config, torch.device("cpu"), Path("."))
    metrics = trainer.train_step(1)
    for values in metrics.values():
        assert torch.isfinite(torch.tensor(values["loss"]))
        assert torch.isfinite(torch.tensor(values["gradient_norm"]))
        assert values["gradient_clipped"] == 0.0


def test_bootstrap_preserving_history_policy_never_evicts_probes() -> None:
    config = replace(
        tiny_config(),
        history_capacity=28,
        bootstrap_actions=8,
        bootstrap_orthogonal=True,
        evaluation_rollout=20,
    )
    state = make_rollout_state(config, 2, 33, torch.device("cpu"))
    bootstrap_before = state.history_before[:, :8].clone()
    bootstrap_action = state.history_action[:, :8].clone()
    policy = HistoryPolicy("bootstrap_8_plus_recent_8", 8, True)
    for _ in range(20):
        before = state.current
        action = torch.randn(2, 8) * 0.01
        after = transition(before, state.matrices, action)
        append_with_policy(state, before, action, after, policy, 8, 28)
    assert torch.equal(state.history_before[:, :8], bootstrap_before)
    assert torch.equal(state.history_action[:, :8], bootstrap_action)
    assert torch.all(state.history_lengths == 16)


def test_all_history_retains_every_real_transition() -> None:
    config = replace(
        tiny_config(),
        history_capacity=20,
        bootstrap_actions=8,
        bootstrap_orthogonal=True,
    )
    state = make_rollout_state(config, 2, 34, torch.device("cpu"))
    policy = HistoryPolicy("all_history", 0, True, True)
    for _ in range(12):
        before = state.current
        action = torch.randn(2, 8) * 0.01
        after = transition(before, state.matrices, action)
        append_with_policy(state, before, action, after, policy, 8, 20)
    assert torch.all(state.history_lengths == 20)


def v42_tiny_config() -> V40Config:
    return replace(
        tiny_config(),
        pool_size=1024,
        batch_size=64,
        history_capacity=16,
        bootstrap_actions=8,
        bootstrap_orthogonal=True,
        training_residual_norm_min=1e-3,
        training_residual_norm_max=1e1,
        training_residual_is_post_bootstrap=True,
        stratified_magnitude_sampling=True,
        convergence_threshold=1e-5,
        magnitude_bin_log10_edges=(-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0),
    )


def test_v42_post_bootstrap_residuals_cover_locked_log_range() -> None:
    config = v42_tiny_config()
    population = ReplayPopulation(config, "query_history", 91)
    norms = population.current.norm(dim=-1)
    assert float(norms.min()) >= config.training_residual_norm_min * (1 - 2e-4)
    assert float(norms.max()) <= config.training_residual_norm_max * (1 + 2e-4)
    assert torch.all(population.magnitude_bin_occupancy() > 0)
    assert population.actual_history_max_error() < 2e-6


def test_v42_stratified_replay_balances_distinct_real_states() -> None:
    config = v42_tiny_config()
    population = ReplayPopulation(config, "query_history", 92)
    indices = population.sample_indices(config.batch_size)
    counts = torch.bincount(
        population.magnitude_bin_ids(population.current[indices]), minlength=8
    )
    assert indices.unique().numel() == config.batch_size
    assert counts.tolist() == [8] * 8


def test_v42_per_bin_diagnostics_are_finite_and_complete() -> None:
    config = replace(
        v42_tiny_config(),
        pool_size=512,
        log_every=1,
        loss_kind="softplus_progress",
        action_regularization=0.0,
        gradient_clip=0.0,
        learning_rate=1e-4,
    )
    trainer = PopulationTrainer(config, torch.device("cpu"), Path("."))
    trainer.train_step(1)
    for arm, rows in trainer.latest_bin_rows.items():
        assert arm in {"query_history", "current_only", "ar_no_fresh"}
        assert len(rows) == 8
        assert [row["sample_count"] for row in rows] == [8] * 8
        for row in rows:
            for key in (
                "q_mean",
                "action_norm_mean",
                "action_gradient_norm",
                "contraction_fraction",
                "history_length_mean",
            ):
                assert torch.isfinite(torch.tensor(row[key]))


def test_v43_harvest_contains_only_real_full_history_states() -> None:
    base = replace(
        tiny_config(),
        history_capacity=6,
        bootstrap_actions=2,
        evaluation_rollout=4,
    )
    experiment = replace(
        V43Config(),
        harvest_worlds=8,
        harvest_batch_size=4,
        harvest_rollout=4,
        harvest_before_minimum=1,
        harvest_after_minimum=2,
        harvest_train_fraction=0.5,
    )
    model = PopulationController("query_history", base)
    model.eval()
    archive, rows = harvest_convergence_pool(
        experiment, base, model, torch.device("cpu")
    )
    assert len(rows) == experiment.harvest_worlds
    assert archive["current"].shape[0] > experiment.harvest_worlds
    assert archive["history_before"].shape[1] == 6
    assert archive_history_error(archive) < 2e-6
    assert set(archive["relation"].tolist()).issubset({0, 1, 2, 3})


def test_v43_mixed_replay_hides_logging_labels_and_world_matrix() -> None:
    base = replace(
        tiny_config(),
        history_capacity=6,
        bootstrap_actions=2,
        evaluation_rollout=4,
    )
    experiment = replace(
        V43Config(),
        harvest_worlds=8,
        harvest_batch_size=4,
        harvest_rollout=4,
        harvest_before_minimum=1,
        harvest_after_minimum=2,
        harvest_train_fraction=0.5,
    )
    model = PopulationController("query_history", base)
    model.eval()
    archive, _ = harvest_convergence_pool(
        experiment, base, model, torch.device("cpu")
    )
    ordinary = ReplayPopulation(base, "query_history", 101).state_dict()
    replay = MixedReplay(experiment, base, archive, ordinary)
    batch, matrices, source, logging = replay.sample(torch.device("cpu"))
    assert int(source.sum()) == 45
    assert int((~source).sum()) == 19
    assert set(batch.__dict__) == {
        "current",
        "previous_action",
        "history",
        "history_lengths",
    }
    assert not batch.history.requires_grad
    assert not matrices.requires_grad
    assert set(logging) == {"depth", "relation"}


def test_v43_posttraining_step_is_finite_and_source_separated(tmp_path: Path) -> None:
    base = replace(
        tiny_config(),
        history_capacity=6,
        bootstrap_actions=2,
        evaluation_rollout=4,
        loss_kind="softplus_progress",
        progress_tau=0.95,
        exploration_std=0.02,
    )
    experiment = replace(
        V43Config(),
        harvest_worlds=8,
        harvest_batch_size=4,
        harvest_rollout=4,
        harvest_before_minimum=1,
        harvest_after_minimum=2,
        harvest_train_fraction=0.5,
    )
    original = PopulationController("query_history", base)
    original.eval()
    archive, _ = harvest_convergence_pool(
        experiment, base, original, torch.device("cpu")
    )
    ordinary = ReplayPopulation(base, "query_history", 102).state_dict()
    replay = MixedReplay(experiment, base, archive, ordinary)
    model = PopulationController("query_history", base)
    trainer = PostTrainer(
        experiment,
        base,
        model,
        original,
        replay,
        torch.device("cpu"),
        tmp_path,
    )
    rows = trainer.step(1)
    assert {row["source"] for row in rows} == {"convergence", "ordinary"}
    for row in rows:
        for key in (
            "loss",
            "q_median",
            "action_norm_median",
            "action_residual_ratio_median",
            "gradient_norm",
            "source_action_gradient_norm",
        ):
            assert torch.isfinite(torch.tensor(row[key]))


def v44_tiny_config() -> V44Config:
    return replace(
        V44Config(),
        seeds=(11,),
        ordinary_pool_size=64,
        recovery_pool_size=32,
        recovery_history_capacity=6,
        recovery_harvest_rollout=4,
        recovery_before_minimum=1,
        recovery_after_minimum=2,
        recovery_refresh_every=10,
        recovery_refresh_count=8,
        batch_size=8,
        training_steps=20,
        smoke_step=5,
        log_every=1,
        checkpoint_every=5,
        small_model_dimension=32,
        small_heads=4,
        small_layers=1,
        small_ffn=64,
        large_model_dimension=48,
        large_heads=4,
        large_layers=2,
        large_ffn=96,
    )


def test_v44_capacity_arm_only_changes_controller_capacity() -> None:
    base = tiny_config()
    experiment = V44Config()
    small = arm_config(base, experiment, "small", 11)
    large = arm_config(base, experiment, "large", 11)
    assert small.residual_dimension == large.residual_dimension == base.residual_dimension
    assert small.action_dimension == large.action_dimension == base.action_dimension
    small_count = parameter_count(PopulationController("query_history", small))
    large_count = parameter_count(PopulationController("query_history", large))
    assert small_count == 71304
    assert 2.0 <= large_count / small_count <= 4.0


def test_v44_recovery_population_is_genuine_and_evolves() -> None:
    base = replace(tiny_config(), bootstrap_actions=2, history_capacity=6)
    experiment = v44_tiny_config()
    config = arm_config(base, experiment, "small", 11)
    model = PopulationController("query_history", config)
    population = RecoveryPopulation(
        experiment, config, model, 11, torch.device("cpu")
    )
    assert population.actual_history_max_error() < 2e-6
    indices = population.sample_indices(3)
    before = population.current[indices].clone()
    action = torch.zeros(3, config.action_dimension)
    after = transition(before, population.matrices[indices], action)
    lengths = population.history_lengths[indices].clone()
    population.append(indices, before, action, after)
    assert torch.equal(
        population.history_lengths[indices],
        (lengths + 1).clamp_max(experiment.recovery_history_capacity),
    )
    assert population.actual_history_max_error() < 2e-6


def test_v44_joint_local_step_separates_both_regimes() -> None:
    base = replace(
        tiny_config(),
        bootstrap_actions=2,
        history_capacity=6,
        loss_kind="softplus_progress",
        progress_tau=0.95,
    )
    experiment = v44_tiny_config()
    runtime = ArmRuntime(
        experiment, base, "small", 11, torch.device("cpu")
    )
    overall, regimes = runtime.train_step(1)
    assert torch.isfinite(torch.tensor(overall["loss"]))
    assert {row["regime"] for row in regimes} == {"ordinary", "convergence"}
    for row in regimes:
        assert torch.isfinite(torch.tensor(row["q_median"]))
        assert torch.isfinite(torch.tensor(row["source_action_gradient_norm"]))
    assert runtime.recovery.actual_history_max_error() < 2e-6


def test_v44_checkpoint_restores_training_state_and_rngs() -> None:
    base = replace(
        tiny_config(),
        bootstrap_actions=2,
        history_capacity=6,
        loss_kind="softplus_progress",
        progress_tau=0.95,
    )
    experiment = v44_tiny_config()
    original = ArmRuntime(experiment, base, "small", 11, torch.device("cpu"))
    original.train_step(1)
    state = original.state_dict()
    restored = ArmRuntime(experiment, base, "small", 11, torch.device("cpu"))
    audit = restored.load_checkpoint_state(state, 1)
    assert audit["action_generator_exactly_saved"]
    assert audit["sample_generator_exactly_saved"]
    for name, value in original.model.state_dict().items():
        assert torch.equal(value, restored.model.state_dict()[name])
    assert torch.equal(original.ordinary.current, restored.ordinary.current)
    assert torch.equal(original.recovery.current, restored.recovery.current)


def test_v45_collection_and_replay_use_only_real_detached_histories() -> None:
    config = replace(
        tiny_config(),
        bootstrap_actions=2,
        history_capacity=6,
        exploration_std=0.0,
    )
    experiment = replace(
        V45Config(),
        max_history=8,
        collection_worlds=6,
        collection_batch_size=3,
        collection_depths=(2, 3),
        batch_size=6,
        latest_batch=3,
        older_batch=2,
        anchor_batch=1,
    )
    model = PopulationController("query_history", config)
    collection = collect_current_policy(
        experiment,
        config,
        model,
        policy_version=0,
        device=torch.device("cpu"),
    )
    assert collection.state_count == 21
    assert collection.history_error() < 2e-6
    anchor = ReplayPopulation(config, "v45_test", 51).state_dict()
    batch, matrices, metadata = mixed_replay_batch(
        experiment,
        config,
        collection,
        [],
        anchor,
        torch.Generator().manual_seed(52),
        torch.device("cpu"),
    )
    assert batch.current.shape == (6, config.residual_dimension)
    assert matrices.shape == (6, config.residual_dimension, config.action_dimension)
    assert set(metadata["source"].tolist()) == {0, 1, 2}
    assert not batch.current.requires_grad
    assert not batch.history.requires_grad
