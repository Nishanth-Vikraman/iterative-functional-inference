from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from torch import Tensor

from ..functions import Functional


@dataclass(frozen=True)
class FunctionalUpdate:
    """Algorithm-neutral description of one proposed function-space update."""

    current_function: Functional
    updated_function: Callable[[Tensor], Tensor]
    metadata: dict[str, Any] = field(default_factory=dict)

    def current(self, x: Tensor) -> Tensor:
        return self.current_function(x)

    def teacher(self, x: Tensor) -> Tensor:
        return self.updated_function(x)

    def displacement(self, x: Tensor) -> Tensor:
        return self.teacher(x) - self.current(x)


class Teacher(ABC):
    """Swappable one-step teacher; IFI only consumes FunctionalUpdate."""

    @abstractmethod
    def propose_update(
        self,
        current_function: Functional,
        support_set: tuple[Tensor, Tensor],
        probe_inputs: Tensor,
    ) -> FunctionalUpdate:
        raise NotImplementedError

