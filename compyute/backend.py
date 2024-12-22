"""Compyute engine utilities."""

from __future__ import annotations

import gc
from abc import ABC
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from types import ModuleType
from typing import Any, ClassVar, Optional, TypeAlias

import cupy
import numpy

__all__ = ["cpu", "cuda", "Device", "CUDA", "CPU", "set_default_device", "use_device"]


class CUDARuntimeError(Exception):
    """Cuda error."""

    def __init__(self, message: str = "invalid device ordinal.") -> None:
        super().__init__(message)


@dataclass(eq=False, repr=False, frozen=True)
class Device(ABC):
    """Device base class."""

    t: str
    index: int = 0
    module: ClassVar[ModuleType] = numpy

    def __eq__(self, other: Any) -> bool:
        return repr(self) == repr(other)

    def __repr__(self) -> str:
        return f"Device({self.name})"

    def __format__(self, __format_spec: str) -> str:
        return self.__repr__().__format__(__format_spec)

    def __enter__(self) -> None: ...

    def __exit__(self, *args: Any) -> None: ...

    @property
    def name(self) -> str:
        return f"{self.t}:{self.index}"


@dataclass(eq=False, repr=False, frozen=True)
class CPU(Device):
    """CPU device."""


@dataclass(eq=False, repr=False, frozen=True)
class CUDA(Device):
    """GPU device."""

    module: ClassVar[ModuleType] = cupy

    def __enter__(self) -> None:
        return self.cupy_device.__enter__()

    def __exit__(self, *args: Any) -> None:
        return self.cupy_device.__exit__(*args)

    @property
    def cupy_device(self) -> cupy.cuda.Device:
        return cupy.cuda.Device(self.index)

    @property
    def properties(self) -> Optional[dict[str, Any]]:
        try:
            return cupy.cuda.runtime.getDeviceProperties(self.index)
        except Exception:
            raise CUDARuntimeError()

    @property
    def memory_info(self) -> dict[str, int]:
        try:
            free, total = self.cupy_device.mem_info
            return {"used": total - free, "free": free, "total": total}
        except Exception:
            raise CUDARuntimeError()


cpu = CPU("cpu")
cuda = CUDA("cuda", 0)


ArrayLike: TypeAlias = numpy.ndarray | cupy.ndarray


def data_to_device(data: ArrayLike, device: Device) -> ArrayLike:
    """Moves the data to the specified device."""
    if device == cpu:
        return cupy.asnumpy(data)
    return cupy.asarray(data)


def gpu_available() -> bool:
    """Checks if GPU is available."""
    try:
        return cupy.cuda.is_available()
    except Exception:
        return False


def get_device_count() -> int:
    """Returns the number of available devices."""
    if not gpu_available():
        return 0
    return cupy.cuda.runtime.getDeviceCount()


def free_memory() -> None:
    """Frees unused blocks from the GPU memory."""
    gc.collect()
    if not gpu_available():
        return
    cupy.get_default_memory_pool().free_all_blocks()


def synchronize() -> None:
    """Synchronizes devices to the current thread."""
    if not gpu_available():
        return
    cupy.cuda.runtime.deviceSynchronize()


def show_cuda_config() -> None:
    """Prints the CUDA configuration."""
    if not gpu_available():
        return
    cupy.show_config()


fallback_default_device: Device = cpu
default_device: Optional[Device] = None


def get_default_device() -> Optional[Device]:
    """Returns the default device."""
    return default_device


def set_default_device(device: Optional[Device]) -> None:
    """Sets the default device for new tensors.

    Parameters
    ----------
    device : Device, optional
        The device to use. Defaults to ``None``.
    """
    global default_device
    default_device = device


@contextmanager
def use_device(device: Device) -> Generator:
    """Context manager to set the default device for new tensors."""
    set_default_device(device)
    try:
        yield
    finally:
        set_default_device(None)


def get_device_from_array(array: ArrayLike) -> Device:
    """Infers the device by type."""
    if isinstance(array, cupy.ndarray):
        return cuda
    return cpu


def select_device(device: Optional[Device]) -> Device:
    """Selects the device. Returns the default device if device is ``None``."""
    return device or default_device or fallback_default_device


class DeviceError(Exception):
    """Tensors with mismatching devices."""
