"""Neural network base module class."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator
from functools import wraps
from typing import Any, Optional

from ...backend import Device, cpu, free_memory
from ...tensor_ops.unary_ops import is_nan
from ...tensors import Tensor
from ...typing import DType
from ...utils import get_debug_mode
from ..functional.functions import FunctionContext, PseudoContext
from ..parameter import Buffer, Parameter

__all__ = ["Module", "Identity", "ModuleList"]


class Module(ABC):
    """Neural network base module.

    Parameters
    ----------
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.
    """

    def __init__(self, label: Optional[str] = None) -> None:
        self.label = label or self.__class__.__name__
        self.function_ctx = FunctionContext()
        self.x: Optional[Tensor] = None
        self.y: Optional[Tensor] = None
        self._is_training = True
        self._retain_values = False
        self._trainable = True
        self._parameters: OrderedDict[str, Parameter] = OrderedDict()
        self._buffers: OrderedDict[str, Buffer] = OrderedDict()
        self._modules: OrderedDict[str, Module] = OrderedDict()

    # ----------------------------------------------------------------------------------
    # PROPERTIES
    # ----------------------------------------------------------------------------------

    @property
    def device(self) -> Device:
        """Device module parameters and variables are stored on."""
        try:
            return next(self.get_parameters()).device
        except StopIteration as e:
            raise ValueError("Module has no parameters.") from e

    def to_device(self, device: Device) -> None:
        """Moves module parameters and variables to the specified device.

        Parameters
        ----------
        device : Device
            Device to move the module parameters and variables to.
        """
        for t in vars(self).values():
            if isinstance(t, Tensor):
                t.ito_device(device)

        for module in self.get_modules(recursive=False):
            module.to_device(device)

    @property
    def dtype(self) -> DType:
        """Data type of module parameters and variables."""
        try:
            return next(self.get_parameters()).dtype
        except StopIteration as e:
            raise ValueError("Module has no parameters.") from e

    def to_type(self, dtype: DType) -> None:
        """Casts module parameters and variables to the specified dtype.

        Parameters
        ----------
        dtype : DType
            DType to cast module parameters and variables to.
        """
        for t in vars(self).values():
            if isinstance(t, Tensor):
                t.ito_type(dtype)

        for module in self.get_modules(recursive=False):
            module.to_type(dtype)

    @property
    def retain_values(self) -> bool:
        """Whether the module should retain intermediate values such as outputs and gradients."""
        return self._retain_values

    @retain_values.setter
    def retain_values(self, value: bool) -> None:
        self._retain_values = value
        for module in self.get_modules(recursive=False):
            module.retain_values = value

    @property
    def trainable(self) -> bool:
        """Whether the module parameters are trainable."""
        return self._trainable

    @trainable.setter
    def trainable(self, value: bool) -> None:
        self._trainable = value
        for module in self.get_modules(recursive=False):
            module.trainable = value

    @property
    def is_training(self) -> bool:
        """Whether the module is in training mode."""
        return self._is_training

    def training(self) -> None:
        """Puts the module in training mode."""
        self._is_training = True
        self.function_ctx = FunctionContext()

        for module in self.get_modules(recursive=False):
            module.training()

    def inference(self) -> None:
        """Puts the module in inference mode."""
        self._is_training = False
        self.function_ctx = PseudoContext()

        for module in self.get_modules(recursive=False):
            module.inference()

    @property
    def n_modules(self) -> int:
        """Number of child modules."""
        return len(list(self.get_modules(recursive=False)))

    # ----------------------------------------------------------------------------------
    # MAGIC METHODS
    # ----------------------------------------------------------------------------------

    def __repr__(self) -> str:
        attrs = [f"{a}={v}" for a, v in vars(self).items() if _is_repr_attr(a, v)]
        repr_string = f"{self.label}(" + ", ".join(attrs) + ")"
        for module in self.get_modules(recursive=False):
            repr_string += "\n" + repr(module)
        return repr_string

    def __bool__(self) -> bool:
        return True

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Buffer):
            self._buffers[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        elif isinstance(value, ModuleList):
            for i, m in enumerate(value):
                self._modules[name + "." + str(i)] = m
        return super().__setattr__(name, value)

    # ----------------------------------------------------------------------------------
    # OTHER OPERATIONS
    # ----------------------------------------------------------------------------------

    def get_modules(self, recursive: bool = True) -> Iterator[Module]:
        """List of child modules.

        Returns
        -------
        Iterator[Module]
            Child modules.
        """
        for m in self._modules.values():
            yield m
            if recursive:
                yield from m.get_modules()

    def get_parameters(self, recursive: bool = True) -> Iterator[Parameter]:
        """Returns an Iterator of module parameters.

        Parameters
        ----------
        recursive : bool, optional
            Whether to include child modules. Defaults to ``True``.

        Returns
        -------
        Iterator[Parameter]
            Iterator of parameters.
        """
        for p in self._parameters.values():
            yield p
        if recursive:
            for m in self.get_modules():
                yield from m.get_parameters(recursive=False)

    def get_buffers(self, recursive: bool = True) -> Iterator[Buffer]:
        """Returns an Iterator of module buffers.

        Parameters
        ----------
        recursive : bool, optional
            Whether to include child modules. Defaults to ``True``.

        Returns
        -------
        Iterator[Buffer]
            Iterator of buffers.
        """
        for b in self._buffers.values():
            yield b
        if recursive:
            for m in self.get_modules():
                yield from m.get_buffers(recursive=False)

    def _get_pointer_state_dict(self) -> OrderedDict[str, Tensor]:
        """Returns a state dict containing pointers to module parameters and buffers."""
        state_dict: OrderedDict[str, Tensor] = OrderedDict()
        state_dict.update(self._parameters)
        state_dict.update(self._buffers)

        for k, m in self._modules.items():
            # get child module state dict
            m_state_dict = m._get_pointer_state_dict()

            # update child module state dict keys
            new_m_state_dict = OrderedDict()
            for key, value in m_state_dict.items():
                new_key = k + "." + key
                new_m_state_dict[new_key] = value

            # update state dict with child module state dict
            state_dict.update(new_m_state_dict)

        return state_dict

    def get_state_dict(self) -> OrderedDict[str, Tensor]:
        """Returns a state dict containing module parameters and buffers.

        Returns
        -------
        OrderedDict[str, Tensor]
            State dict containing parameters and buffers.
        """
        state_dict = self._get_pointer_state_dict()

        for k, v in state_dict.items():
            state_dict[k] = v.to_cpu()

        return state_dict

    def load_state_dict(
        self, state_dict: OrderedDict, target_device: Optional[Device] = cpu
    ) -> None:
        """Loads the module state from a state dict.

        Parameters
        ----------
        state_dict : OrderedDict
            State dict containing parameters and buffers.
        target_device : Device, optional
            Device to move the parameters and buffers to. Defaults to ``cpu``.
        """
        self_state_dict = self._get_pointer_state_dict()
        for (k1, v1), (k2, v2) in zip(self_state_dict.items(), state_dict.items()):
            if k1 != k2:
                raise ValueError(f"State dict key mismatch: {k1},  {k2}")

            v1.data = v2.to_device(target_device).data
            if v2.grad is not None:
                v1.grad = v2.grad.to_device(target_device)

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the module.

        Parameters
        ----------
        x : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Output tensor.
        """

    @abstractmethod
    def backward(self, dy: Tensor) -> Tensor:
        """Backward pass of the module.

        Parameters
        ----------
        dy : Tensor
            Output gradient tensor.

        Returns
        -------
        Tensor
            Input gradient tensor.
        """

    @staticmethod
    def register_forward(fwd_fn: Callable) -> Callable:
        """Registers a the forward method to the module."""

        @wraps(fwd_fn)
        def wrapper(m: Module, x: Tensor) -> Tensor:

            m.function_ctx.clear()

            if get_debug_mode():
                dt = time.perf_counter()
                y = fwd_fn(m, x)
                dt = (time.perf_counter() - dt) * 1e3
                print(_format_debug_str(m.label, "fwd", x.dtype, y.dtype, dt))
            else:
                y = fwd_fn(m, x)

            assert not is_nan(y).any().item(), "NaNs detected in " + repr(m)

            if m.retain_values:
                m.x = x
                m.y = y

            return y

        return wrapper

    @staticmethod
    def register_backward(bwd_fn: Callable) -> Callable:
        """Registers a the backward method for the module."""

        @wraps(bwd_fn)
        def wrapper(m: Module, dy: Tensor) -> Tensor:
            if not m.is_training:
                raise AttributeError(f"{m.label} is not in training mode.")

            if get_debug_mode():
                dt = time.perf_counter()
                dx = bwd_fn(m, dy)
                dt = (time.perf_counter() - dt) * 1e3
                print(_format_debug_str(m.label, "bwd", dx.dtype, dy.dtype, dt))
            else:
                dx = bwd_fn(m, dy)

            assert not is_nan(dx).any().item(), "NaNs detected in " + repr(m)
            assert not m.function_ctx.context, "Context memory leak in " + repr(m)

            if m.retain_values and m.x and m.y:
                m.x.grad = dx
                m.y.grad = dy

            return dx

        return wrapper

    def clean(self, force: bool = False) -> None:
        """Cleans up temporary values like outputs and gradients.

        Parameters
        ----------
        force : bool, optional
            Whether to force clean and ignore ``retain_values``. Defaults to ``False``.
        """
        self.function_ctx.context.clear()

        if not self._retain_values or force:
            self.x = self.y = None
            for p in self.get_parameters(recursive=False):
                p.grad = None

        for module in self.get_modules(recursive=False):
            module.clean(force)

        free_memory()

    def update_parameter_grad(
        self, parameter: Optional[Parameter], grad: Optional[Tensor]
    ) -> None:
        """Updates the parameter gradients."""
        if not (self.trainable and parameter and grad):
            return
        if parameter.grad is None:
            parameter.grad = grad
        else:
            parameter.grad += grad  # for gradient accumulation


class Identity(Module):
    """Identity module that just forwards inputs and gradients.

    Parameters
    ----------
    label : str, optional
        Module label. Defaults to ``None``. If ``None``, the class name is used.
    """

    @Module.register_forward
    def forward(self, x: Tensor) -> Tensor:
        return x

    @Module.register_backward
    def backward(self, dy: Tensor) -> Tensor:
        return dy


class ModuleList(list):
    """List of modules.

    Parameters
    ----------
    modules : Iterable[Module]
        Modules to add to the list.
    """

    def __init__(self, modules: Iterable[Module]) -> None:
        super().__init__(modules)


def _is_repr_attr(attr: str, value: Any) -> bool:
    """Checks if an attribute should be included int the class representation."""
    return all(
        [
            attr not in {"label", "function_ctx"},
            not attr.startswith("_"),
            not isinstance(value, (Tensor, Module, ModuleList)),
            value is not None,
        ]
    )


def _format_debug_str(
    label: str, mode: str, in_dtype: DType, out_dtype: DType, dt: float
) -> str:
    """Formats debug string for forward and backward passes."""
    return (
        f"{label:20s} | {mode} | "
        f"{in_dtype:15s} | "
        f"{out_dtype:15s} | "
        f"{dt=:>10.4f} ms"
    )
