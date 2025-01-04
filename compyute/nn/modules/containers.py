"""Neural network container modules."""

from itertools import accumulate
from typing import Optional

from ...tensor_ops.reduction_ops import tensorsum
from ...tensor_ops.shape_ops import concat, split
from ...tensors import Tensor
from .module import Module, ModuleList

__all__ = ["ParallelAdd", "ParallelConcat", "ResidualConnection", "Sequential"]


class Sequential(Module):
    """Container that processes modules sequentially.

    .. math::
        y = f_n( ... f_2(f_1(x)) ... )

    Parameters
    ----------
    *modules : Module
        Modules used in the container.
    label : str, optional
        Container label. Defaults to ``None``. If ``None``, the class name is used.
    """

    def __init__(self, *modules: Module, label: Optional[str] = None) -> None:
        super().__init__(label)
        if not modules:
            raise EmptyContainerError()

        self.layers = ModuleList(modules)

    def add(self, module: Module) -> None:
        """Add a module to the container.

        Parameters
        ----------
        module : Module
            Module to add.
        """

        self.layers.append(module)

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        for module in self.layers:
            x = module(x)
        return x

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        for module in reversed(self.layers):
            dy = module.backward(dy)
        return dy


class ParallelConcat(Module):
    r"""Container that processes modules in parallel and concatenates their outputs.

    .. math::
        y = \text{concatenate}(f_1(x), f_2(x), ..., f_n(x))

    Parameters
    ----------
    *modules : Module
        Modules used in the parallel container.
    concat_dim : int, optional
        Dimension along which the output of the parallel modules
        shall be concatinated. Defaults to ``-1``.
    label : str, optional
        Container label. Defaults to ``None``. If ``None``, the class name is used.
    """

    def __init__(
        self, *modules: Module, concat_dim: int = -1, label: Optional[str] = None
    ) -> None:
        super().__init__(label)
        if not modules:
            raise EmptyContainerError()

        self.modules = ModuleList(modules)
        self.concat_dim = concat_dim

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        ys = [m(x) for m in self.modules]
        split_indices = list(accumulate(y.shape[self.concat_dim] for y in ys[:-1]))
        y = concat(ys, dim=self.concat_dim)
        self.ctx.add(split_indices)
        return y

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        split_indices = self.ctx.get()
        splits = split(dy, splits=split_indices, dim=self.concat_dim)
        return tensorsum(m.backward(s) for m, s in zip(self.modules, splits))


class ParallelAdd(Module):
    r"""Container that processes modules in parallel and sums their outputs element-wise.

    .. math::
        y = \sum_{i=1}^N f_i(x)

    Parameters
    ----------
    *modules : Module
        Modules used in the container.
    label : str, optional
        Container label. Defaults to ``None``. If ``None``, the class name is used.
    training : bool, optional
        Whether the container and its modules should be in training mode. Defaults to ``False``.
    """

    def __init__(self, *modules: Module, label: Optional[str] = None) -> None:
        super().__init__(label)
        if not modules:
            raise EmptyContainerError()
        self.modules = ModuleList(modules)

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        return tensorsum(m(x) for m in self.modules)

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        return tensorsum(m.backward(dy) for m in self.modules)


class ResidualConnection(Module):
    """Residual container implementing a residual connection around a block of modules.
    Modules in the residual block are processed sequentially.

    .. math::
        y = x + f(x)

    Parameters
    ----------
    *modules : Module
        Modules used in the residual block. They are processed sequentially.
    residual_projection : Module, optional
        Module used as a projection in the residual pathway to achieve matching dimensions.
        Defaults to ``None``. Using a projection within the residual pathway should be avoided,
        and instead projections should be part of the residual block.
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.
    """

    def __init__(
        self,
        *modules: Module,
        residual_proj: Optional[Module] = None,
        label: Optional[str] = None
    ) -> None:
        if not modules:
            raise EmptyContainerError()
        super().__init__(label)

        self.residual_block = modules[0] if len(modules) == 1 else Sequential(*modules)
        self.residual_proj = residual_proj

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        y = self.residual_block(x)
        y += self.residual_proj(x) if self.residual_proj else x
        return y

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        dx = self.residual_block.backward(dy)
        dx += self.residual_proj.backward(dy) if self.residual_proj else dy
        return dx


class EmptyContainerError(Exception):
    """Exception for empty containers."""

    def __init__(self, message: str = "At least one module is required.") -> None:
        super().__init__(message)
