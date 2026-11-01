from dataclasses import replace

import torch

from ifi.v40_config import V40Config
from ifi.v40_models import ControllerBatch
from ifi.v40_models import PopulationController
from ifi.v45_replay import RawReplayBatch
from ifi.v46w_config import V46WConfig
from ifi.v46w_experiment import (
    empirical_history_prediction,
    collect_datasets,
    evaluate_world_model,
    fit_global_response,
    mixed_action_kinds,
    normalized_delta_loss,
    train_to_step,
)
from ifi.v46w_model import ResidualWorldModel, WorldModelBatch


def tiny_configs() -> tuple[V40Config, V46WConfig]:
    base = replace(
        V40Config(),
        residual_dimension=4,
        action_dimension=4,
        model_dimension=16,
        attention_heads=2,
        attention_layers=1,
        ffn_dimension=32,
    )
    experiment = replace(
        V46WConfig(), model_dimension=16, heads=2, layers=1, ffn_dimension=32
    )
    return base, experiment


def test_world_model_predicts_vector_and_has_candidate_action_gradient() -> None:
    base, experiment = tiny_configs()
    model = ResidualWorldModel(base, experiment)
    action = torch.randn(3, 4, requires_grad=True)
    batch = WorldModelBatch(
        ControllerBatch(
            current=torch.randn(3, 4),
            previous_action=torch.randn(3, 4),
            history=torch.randn(3, 5, 18),
            history_lengths=torch.tensor([5, 3, 0]),
        ),
        action,
    )
    prediction = model(batch)
    assert prediction.shape == (3, 4)
    prediction.square().mean().backward()
    assert action.grad is not None
    assert float(action.grad.abs().sum()) > 0
    assert all("position" not in name for name, _ in model.named_parameters())


def test_normalized_delta_loss_is_zero_at_exact_prediction() -> None:
    target = torch.randn(7, 4)
    loss = normalized_delta_loss(target.clone(), target, 1e-3)
    assert torch.equal(loss, torch.zeros_like(loss))


def test_candidate_mixture_is_locked_40_20_20_20() -> None:
    _, experiment = tiny_configs()
    kinds = mixed_action_kinds(100, experiment)
    assert [(kinds == index).sum().item() for index in range(4)] == [40, 20, 20, 20]


def test_history_least_squares_recovers_observed_linear_world() -> None:
    generator = torch.Generator().manual_seed(46)
    batch, dimension, history = 3, 4, 8
    matrix = torch.randn(batch, dimension, dimension, generator=generator)
    actions = torch.randn(batch, history, dimension, generator=generator)
    before = torch.randn(batch, history, dimension, generator=generator)
    after = before + torch.bmm(
        matrix.repeat_interleave(history, dim=0),
        actions.reshape(-1, dimension, 1),
    ).reshape(batch, history, dimension)
    candidate = torch.randn(batch, dimension, generator=generator)
    raw = RawReplayBatch(
        matrices=matrix,
        current=torch.randn(batch, dimension, generator=generator),
        previous_action=torch.zeros(batch, dimension),
        history_before=before,
        history_action=actions,
        history_after=after,
        history_lengths=torch.full((batch,), history),
        depth=torch.zeros(batch, dtype=torch.int64),
        policy_version=torch.zeros(batch, dtype=torch.int64),
        source=torch.zeros(batch, dtype=torch.int64),
    )
    predicted = empirical_history_prediction(raw, candidate)
    expected = torch.bmm(matrix, candidate.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(predicted, expected, atol=2e-5, rtol=2e-5)


def test_tiny_world_model_pipeline_runs_end_to_end() -> None:
    base, experiment = tiny_configs()
    experiment = replace(
        experiment,
        max_history=8,
        train_worlds=8,
        validation_worlds=4,
        heldout_worlds=4,
        ood_worlds=4,
        collection_depths=(2, 3),
        collection_batch_size=4,
        batch_size=8,
        evaluation_states=4,
    )
    controller = PopulationController("query_history", base).eval().requires_grad_(False)
    collections = collect_datasets(
        experiment, base, controller, torch.device("cpu")
    )
    model = ResidualWorldModel(base, experiment)
    rows, _ = train_to_step(
        experiment,
        base,
        model,
        collections["train"],
        seed=11,
        start_step=0,
        end_step=2,
        device=torch.device("cpu"),
    )
    assert rows
    metrics = evaluate_world_model(
        experiment,
        base,
        model,
        collections,
        fit_global_response(collections["train"]),
        seed=11,
        device=torch.device("cpu"),
    )
    assert metrics
    assert {row["action_kind"] for row in metrics} == {
        "t_proposal",
        "scaled_t",
        "random",
        "orthogonal",
    }
