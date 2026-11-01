import torch

from ifi.config import OperatorConfig
from ifi.functions import ComposedFunction
from ifi.operators import build_function_system, project_update
from ifi.teachers import FunctionalUpdate


def test_phi_zero_is_exact_identity():
    config = OperatorConfig(dimension=12, action_dimension=4, hidden_dimension=16, basis_depth=1)
    system, operator = build_function_system(config, torch.device("cpu"))
    function = ComposedFunction(system, operator)
    x = torch.linspace(-2, 2, 17).unsqueeze(-1)
    zero = torch.zeros(config.action_dimension)
    torch.testing.assert_close(function.append(zero)(x), function(x), rtol=0, atol=0)


def test_projection_recovers_an_in_span_action():
    torch.manual_seed(1)
    config = OperatorConfig(
        dimension=12, action_dimension=4, hidden_dimension=24, basis_depth=1,
        projection_ridge=1e-8,
    )
    system, operator = build_function_system(config, torch.device("cpu"))
    current = ComposedFunction(system, operator)
    expected_phi = torch.tensor([0.2, -0.4, 0.1, 0.3])
    teacher_function = current.append(expected_phi)
    update = FunctionalUpdate(current, teacher_function)
    probes = torch.linspace(-2, 2, 128).unsqueeze(-1)
    projection = project_update(operator, current, update, probes, config.projection_ridge)
    torch.testing.assert_close(projection.phi, expected_phi, atol=2e-4, rtol=2e-4)
    assert projection.relative_residual.item() < 1e-7


def test_freeze_is_permanent_and_observable():
    config = OperatorConfig(dimension=12, action_dimension=4, hidden_dimension=16, basis_depth=1)
    _, operator = build_function_system(config, torch.device("cpu"))
    assert not operator.frozen
    operator.freeze()
    assert operator.frozen

