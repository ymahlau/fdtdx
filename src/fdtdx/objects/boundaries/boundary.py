from abc import ABC, abstractmethod

from fdtdx.core.jax.pytrees import ExtendedTreeClass, extended_autoinit


@extended_autoinit
class BaseBoundaryState(ExtendedTreeClass):
    pass


@extended_autoinit
class BaseBoundary(NoMaterial, ABC):
    @abstractmethod
    def init_state(
        self,
    ) -> BaseBoundaryState:
        raise NotImplementedError()

    @abstractmethod
    def reset_state(self, state: BaseBoundaryState) -> BaseBoundaryState:
        raise NotImplementedError()
