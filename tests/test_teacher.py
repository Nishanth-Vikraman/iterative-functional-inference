import torch

from ifi.config import TeacherConfig
from ifi.teachers.rkhs import RKHSFunctionalGradientTeacher, rbf_kernel


class LinearFunction:
    def __call__(self, x):
        return 0.25 * x


def test_rkhs_teacher_is_exact_one_step_and_probe_independent():
    config = TeacherConfig(bandwidth=0.7, step_size=0.6, normalize_by_support=True)
    teacher = RKHSFunctionalGradientTeacher(config)
    x = torch.tensor([[-1.0], [0.5], [1.0]])
    y = torch.tensor([[0.4], [-0.2], [1.2]])
    current = LinearFunction()
    update_a = teacher.propose_update(current, (x, y), torch.tensor([[99.0]]))
    update_b = teacher.propose_update(current, (x, y), torch.tensor([[-99.0], [4.0]]))
    query = torch.linspace(-1.5, 1.5, 9).unsqueeze(-1)
    residual = y - current(x)
    expected = current(query) + rbf_kernel(query, x, config.bandwidth) @ (
        config.step_size * residual / len(x)
    )
    torch.testing.assert_close(update_a.teacher(query), expected)
    torch.testing.assert_close(update_a.teacher(query), update_b.teacher(query))


def test_functional_update_exposes_current_teacher_and_displacement():
    teacher = RKHSFunctionalGradientTeacher(TeacherConfig())
    x = torch.tensor([[-1.0], [1.0]])
    y = torch.tensor([[1.0], [-1.0]])
    update = teacher.propose_update(LinearFunction(), (x, y), x)
    torch.testing.assert_close(update.displacement(x), update.teacher(x) - update.current(x))

