from dataclasses import replace

import torch

from ifi.v40_config import V40Config
from ifi.v40_models import ControllerBatch, PopulationController
from ifi.v45_config import V45Config
from ifi.v45_replay import collect_current_policy
from ifi.v45q_config import V45QConfig
from ifi.v45q_model import ContractionCritic, CriticBatch


def tiny_base() -> V40Config:
    return replace(
        V40Config(),
        residual_dimension=4,
        action_dimension=4,
        model_dimension=16,
        attention_heads=2,
        attention_layers=1,
        ffn_dimension=32,
        bootstrap_actions=2,
        history_capacity=4,
        evaluation_batch_size=4,
    )


def test_contraction_critic_has_action_gradient_but_no_world_input() -> None:
    base = tiny_base()
    experiment = replace(
        V45QConfig(), q_model_dimension=16, q_heads=2, q_layers=1, q_ffn=32
    )
    critic = ContractionCritic(base, experiment)
    action = torch.randn(3, 4, requires_grad=True)
    batch = CriticBatch(
        ControllerBatch(
            current=torch.randn(3, 4),
            previous_action=torch.randn(3, 4),
            history=torch.randn(3, 5, 18),
            history_lengths=torch.tensor([5, 3, 0]),
        ),
        action,
    )
    log_q, harm = critic(batch)
    (log_q.mean() + harm.mean()).backward()
    assert action.grad is not None
    assert float(action.grad.abs().sum()) > 0
    assert all("position" not in name for name, _ in critic.named_parameters())


def test_capped_collection_retains_only_real_transitions() -> None:
    base = tiny_base()
    experiment = replace(
        V45Config(),
        max_history=4,
        collection_worlds=4,
        collection_batch_size=2,
        collection_depths=(7,),
    )
    model = PopulationController("query_history", base)
    collection = collect_current_policy(
        experiment, base, model, policy_version=0, device=torch.device("cpu")
    )
    assert collection.history_error() < 2e-6
    deepest = collection.batch_from_positions(
        torch.tensor([collection.state_count - 1]), source_id=0
    )
    assert int(deepest.history_lengths[0]) == 4
