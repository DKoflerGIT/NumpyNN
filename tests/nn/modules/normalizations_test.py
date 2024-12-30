"""Normalization module tests"""

import pytest
import torch

from compyute.nn import BatchNorm1D, BatchNorm2D, LayerNorm, RMSNorm
from tests.utils import get_random_floats, is_close

bn1d_testdata = [(8, 16), (8, 16, 32)]
bn2d_testdata = [(8, 16, 32, 32), (16, 32, 64, 64)]
ln_testdata = [
    ((8, 16, 32), (32,)),
    ((8, 16, 32), (16, 32)),
    ((8, 16, 32, 64), (64,)),
    ((8, 16, 32, 64), (32, 64)),
]
rms_testdata = [
    ((8, 16, 32), 32),
    ((8, 16, 32, 64), 64),
]
eps_testdata = [1e-5, 1e-4]
m_testdata = [0.1, 0.2]


@pytest.mark.parametrize("shape", bn1d_testdata)
@pytest.mark.parametrize("eps", eps_testdata)
@pytest.mark.parametrize("m", m_testdata)
def test_batchnorm1d(shape, eps, m) -> None:
    """Test for the batchnorm 1d layer."""

    # init compyute module
    compyute_module = BatchNorm1D(shape[1], eps, m)

    # init torch module
    torch_module = torch.nn.BatchNorm1d(shape[1], eps, m)

    # forward
    compyute_x, torch_x = get_random_floats(shape)
    compyute_y = compyute_module(compyute_x)
    torch_y = torch_module(torch_x)
    assert is_close(compyute_y, torch_y)
    assert is_close(compyute_module.rmean, torch_module.running_mean)
    assert is_close(compyute_module.rvar, torch_module.running_var)

    # backward
    compyute_dy, torch_dy = get_random_floats(compyute_y.shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    assert is_close(compyute_dx, torch_x.grad)
    assert is_close(compyute_module.w.grad, torch_module.weight.grad)
    assert is_close(compyute_module.b.grad, torch_module.bias.grad)


@pytest.mark.parametrize("shape", bn2d_testdata)
@pytest.mark.parametrize("eps", eps_testdata)
@pytest.mark.parametrize("m", m_testdata)
def test_batchnorm2d(shape, eps, m) -> None:
    """Test for the batchnorm 2d layer."""

    # init compyute module
    compyute_module = BatchNorm2D(shape[1], eps, m)

    # init torch module
    torch_module = torch.nn.BatchNorm2d(shape[1], eps, m)

    # forward
    compyute_x, torch_x = get_random_floats(shape)
    compyute_y = compyute_module(compyute_x)
    torch_y = torch_module(torch_x)
    assert is_close(compyute_y, torch_y)
    assert is_close(compyute_module.rmean, torch_module.running_mean)
    assert is_close(compyute_module.rvar, torch_module.running_var)

    # backward
    compyute_dy, torch_dy = get_random_floats(compyute_y.shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    assert is_close(compyute_dx, torch_x.grad)
    assert is_close(compyute_module.w.grad, torch_module.weight.grad)
    assert is_close(compyute_module.b.grad, torch_module.bias.grad)


@pytest.mark.parametrize("shape,normalized_shape", ln_testdata)
@pytest.mark.parametrize("eps", eps_testdata)
def test_layernorm(shape, normalized_shape, eps) -> None:
    """Test for the layernorm layer."""
    # init compyute module
    compyute_module = LayerNorm(normalized_shape, eps)

    # init torch module
    torch_module = torch.nn.LayerNorm(normalized_shape, eps)

    # forward
    compyute_x, torch_x = get_random_floats(shape)
    compyute_y = compyute_module(compyute_x)
    torch_y = torch_module(torch_x)
    assert is_close(compyute_y, torch_y)

    # backward
    compyute_dy, torch_dy = get_random_floats(shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    assert is_close(compyute_dx, torch_x.grad)
    assert is_close(compyute_module.w.grad, torch_module.weight.grad)
    assert is_close(compyute_module.b.grad, torch_module.bias.grad)


@pytest.mark.parametrize("shape,normalized_shape", rms_testdata)
@pytest.mark.parametrize("eps", eps_testdata)
def test_rmsnorm(shape, normalized_shape, eps) -> None:
    """Test for the rmsnorm layer."""

    # init compyute module
    compyute_module = RMSNorm((normalized_shape,), eps=eps)

    # init torch module
    torch_module = torch.nn.RMSNorm(normalized_shape, eps)

    # forward
    compyute_x, torch_x = get_random_floats(shape)
    compyute_y = compyute_module(compyute_x)
    torch_y = torch_module(torch_x)
    assert is_close(compyute_y, torch_y)

    # backward
    compyute_dy, torch_dy = get_random_floats(shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    assert is_close(compyute_dx, torch_x.grad)
    assert is_close(compyute_module.w.grad, torch_module.weight.grad)
