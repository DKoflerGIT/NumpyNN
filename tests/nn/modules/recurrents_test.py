"""Recurrent module tests"""

import pytest
import torch

from compyute.nn import GRU, LSTM, Parameter, Recurrent, Sequential
from compyute.tensor_ops.shape_ops import concat, split
from tests.utils import get_random_floats, get_random_params, is_close

shape_testdata = [(8, 16, 32, 64), (16, 32, 64, 128)]
act_testdata = ["tanh", "relu"]


@pytest.mark.parametrize("shape", shape_testdata)
@pytest.mark.parametrize("act", act_testdata)
def test_recurrent(shape, act) -> None:
    """Test for the recurrent layer."""

    B, X, Cin, Ch = shape
    shape_x = (B, X, Cin)
    shape_w_in_1 = (Ch, Cin)
    shape_b_in_1 = (Ch,)
    shape_w_h_1 = (Ch, Ch)
    shape_b_h_1 = (Ch,)
    shape_w_in_2 = (Ch, Ch)
    shape_b_in_2 = (Ch,)
    shape_w_h_2 = (Ch, Ch)
    shape_b_h_2 = (Ch,)

    # init parameters
    compyute_w_in_1, torch_w_in_1 = get_random_params(shape_w_in_1)
    compyute_b_in_1, torch_b_in_1 = get_random_params(shape_b_in_1)
    compyute_w_h_1, torch_w_h_1 = get_random_params(shape_w_h_1)
    compyute_b_h_1, torch_b_h_1 = get_random_params(shape_b_h_1)
    compyute_w_in_2, torch_w_in_2 = get_random_params(shape_w_in_2)
    compyute_b_in_2, torch_b_in_2 = get_random_params(shape_b_in_2)
    compyute_w_h_2, torch_w_h_2 = get_random_params(shape_w_h_2)
    compyute_b_h_2, torch_b_h_2 = get_random_params(shape_b_h_2)

    # init compyute module
    compyute_recurrent1 = Recurrent(Cin, Ch, activation=act)
    compyute_recurrent2 = Recurrent(Ch, Ch, activation=act)
    compyute_recurrent1.w_i = compyute_w_in_1
    compyute_recurrent1.b_i = compyute_b_in_1
    compyute_recurrent1.w_h = compyute_w_h_1
    compyute_recurrent1.b_h = compyute_b_h_1
    compyute_recurrent2.w_i = compyute_w_in_2
    compyute_recurrent2.b_i = compyute_b_in_2
    compyute_recurrent2.w_h = compyute_w_h_2
    compyute_recurrent2.b_h = compyute_b_h_2
    compyute_module = Sequential(compyute_recurrent1, compyute_recurrent2)

    # init torch module
    torch_module = torch.nn.RNN(Cin, Ch, 2, act, batch_first=True)
    torch_module.weight_ih_l0 = torch_w_in_1
    torch_module.bias_ih_l0 = torch_b_in_1
    torch_module.weight_hh_l0 = torch_w_h_1
    torch_module.bias_hh_l0 = torch_b_h_1
    torch_module.weight_ih_l1 = torch_w_in_2
    torch_module.bias_ih_l1 = torch_b_in_2
    torch_module.weight_hh_l1 = torch_w_h_2
    torch_module.bias_hh_l1 = torch_b_h_2

    # forward
    compyute_x, torch_x = get_random_floats(shape_x)
    compyute_y = compyute_module(compyute_x)
    torch_y = torch_module(torch_x)[0]  # outputs tuple of y and hidden_states
    assert is_close(compyute_y, torch_y)

    # backward
    compyute_dy, torch_dy = get_random_floats(compyute_y.shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)

    assert is_close(compyute_dx, torch_x.grad)
    assert is_close(compyute_w_in_1.grad, torch_module.weight_ih_l0.grad)
    assert is_close(compyute_b_in_1.grad, torch_module.bias_ih_l0.grad)
    assert is_close(compyute_w_h_1.grad, torch_module.weight_hh_l0.grad)
    assert is_close(compyute_b_h_1.grad, torch_module.bias_hh_l0.grad)
    assert is_close(compyute_w_in_2.grad, torch_module.weight_ih_l1.grad)
    assert is_close(compyute_b_in_2.grad, torch_module.bias_ih_l1.grad)
    assert is_close(compyute_w_h_2.grad, torch_module.weight_hh_l1.grad)
    assert is_close(compyute_b_h_2.grad, torch_module.bias_hh_l1.grad)


@pytest.mark.parametrize("shape", shape_testdata)
def test_lstm(shape) -> None:
    """Test for the lstm layer."""
    B, X, Cin, Ch = shape
    shape_x = (B, X, Cin)

    # init parameters
    compyute_w_i_1, torch_w_i_1 = get_random_params((4 * Ch, Cin))
    compyute_b_i_1, torch_b_i_1 = get_random_params((4 * Ch,))
    compyute_w_h_1, torch_w_h_1 = get_random_params((4 * Ch, Ch))
    compyute_b_h_1, torch_b_h_1 = get_random_params((4 * Ch,))
    compyute_w_i_2, torch_w_i_2 = get_random_params((4 * Ch, Ch))
    compyute_b_i_2, torch_b_i_2 = get_random_params((4 * Ch,))
    compyute_w_h_2, torch_w_h_2 = get_random_params((4 * Ch, Ch))
    compyute_b_h_2, torch_b_h_2 = get_random_params((4 * Ch,))

    # init compyute module
    lstm1 = LSTM(Cin, Ch)
    lstm1.w_i = compyute_w_i_1
    lstm1.b_i = compyute_b_i_1
    lstm1.w_h = compyute_w_h_1
    lstm1.b_h = compyute_b_h_1

    lstm2 = LSTM(Ch, Ch)
    lstm2.w_i = compyute_w_i_2
    lstm2.b_i = compyute_b_i_2
    lstm2.w_h = compyute_w_h_2
    lstm2.b_h = compyute_b_h_2

    compyute_module = Sequential(lstm1, lstm2)

    # init torch module
    torch_module = torch.nn.LSTM(Cin, Ch, 2, batch_first=True)
    torch_module.weight_ih_l0 = torch_w_i_1
    torch_module.bias_ih_l0 = torch_b_i_1
    torch_module.weight_hh_l0 = torch_w_h_1
    torch_module.bias_hh_l0 = torch_b_h_1
    torch_module.weight_ih_l1 = torch_w_i_2
    torch_module.bias_ih_l1 = torch_b_i_2
    torch_module.weight_hh_l1 = torch_w_h_2
    torch_module.bias_hh_l1 = torch_b_h_2

    # forward
    compyute_x, torch_x = get_random_floats(shape_x)
    compyute_y = compyute_module(compyute_x)
    torch_y = torch_module(torch_x)[0]  # outputs tuple of y and hidden_states
    assert is_close(compyute_y, torch_y)

    # backward
    compyute_dy, torch_dy = get_random_floats(compyute_y.shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    assert is_close(compyute_dx, torch_x.grad)
    assert is_close(compyute_w_i_1.grad, torch_module.weight_ih_l0.grad)
    assert is_close(compyute_b_i_1.grad, torch_module.bias_ih_l0.grad)
    assert is_close(compyute_w_h_1.grad, torch_module.weight_hh_l0.grad)
    assert is_close(compyute_b_h_1.grad, torch_module.bias_hh_l0.grad)
    assert is_close(compyute_w_i_2.grad, torch_module.weight_ih_l1.grad)
    assert is_close(compyute_b_i_2.grad, torch_module.bias_ih_l1.grad)
    assert is_close(compyute_w_h_2.grad, torch_module.weight_hh_l1.grad)
    assert is_close(compyute_b_h_2.grad, torch_module.bias_hh_l1.grad)


@pytest.mark.parametrize("shape", shape_testdata)
def test_gru(shape) -> None:
    """Test for the gru layer."""

    B, X, Cin, Ch = shape
    shape_x = (B, X, Cin)
    shape_w_i_1 = (3 * Ch, Cin)
    shape_b_i_1 = (3 * Ch,)
    shape_w_h_1 = (3 * Ch, Ch)
    shape_b_h_1 = (3 * Ch,)
    shape_w_i_2 = (3 * Ch, Ch)
    shape_b_i_2 = (3 * Ch,)
    shape_w_h_2 = (3 * Ch, Ch)
    shape_b_h_2 = (3 * Ch,)

    # init parameters
    compyute_w_i_1, torch_w_i_1 = get_random_params(shape_w_i_1)
    compyute_b_i_1, torch_b_i_1 = get_random_params(shape_b_i_1)
    compyute_w_h_1, torch_w_h_1 = get_random_params(shape_w_h_1)
    compyute_b_h_1, torch_b_h_1 = get_random_params(shape_b_h_1)
    compyute_w_i_2, torch_w_i_2 = get_random_params(shape_w_i_2)
    compyute_b_i_2, torch_b_i_2 = get_random_params(shape_b_i_2)
    compyute_w_h_2, torch_w_h_2 = get_random_params(shape_w_h_2)
    compyute_b_h_2, torch_b_h_2 = get_random_params(shape_b_h_2)

    # init compyute module
    gru1 = GRU(Cin, Ch)
    gru1.w_i = compyute_w_i_1
    gru1.b_i = compyute_b_i_1
    gru1.w_h = compyute_w_h_1
    gru1.b_h = compyute_b_h_1

    gru2 = GRU(Ch, Ch)
    gru2.w_i = compyute_w_i_2
    gru2.b_i = compyute_b_i_2
    gru2.w_h = compyute_w_h_2
    gru2.b_h = compyute_b_h_2

    compyute_module = Sequential(gru1, gru2)

    # init torch module
    torch_module = torch.nn.GRU(Cin, Ch, num_layers=2, batch_first=True)
    torch_module.weight_ih_l0 = torch_w_i_1
    torch_module.bias_ih_l0 = torch_b_i_1
    torch_module.weight_hh_l0 = torch_w_h_1
    torch_module.bias_hh_l0 = torch_b_h_1
    torch_module.weight_ih_l1 = torch_w_i_2
    torch_module.bias_ih_l1 = torch_b_i_2
    torch_module.weight_hh_l1 = torch_w_h_2
    torch_module.bias_hh_l1 = torch_b_h_2

    # forward
    compyute_x, torch_x = get_random_floats(shape_x)
    compyute_y = compyute_module(compyute_x)
    torch_y = torch_module(torch_x)[0]  # outputs tuple of y and hidden_states
    assert is_close(compyute_y, torch_y)

    # backward
    compyute_dy, torch_dy = get_random_floats(compyute_y.shape, torch_grad=False)
    compyute_dx = compyute_module.backward(compyute_dy)
    torch_y.backward(torch_dy)
    assert is_close(compyute_dx, torch_x.grad)
    assert is_close(compyute_w_i_1.grad, torch_module.weight_ih_l0.grad)
    assert is_close(compyute_b_i_1.grad, torch_module.bias_ih_l0.grad)
    assert is_close(compyute_w_h_1.grad, torch_module.weight_hh_l0.grad)
    assert is_close(compyute_b_h_1.grad, torch_module.bias_hh_l0.grad)
    assert is_close(compyute_w_i_2.grad, torch_module.weight_ih_l1.grad)
    assert is_close(compyute_b_i_2.grad, torch_module.bias_ih_l1.grad)
    assert is_close(compyute_w_h_2.grad, torch_module.weight_hh_l1.grad)
    assert is_close(compyute_b_h_2.grad, torch_module.bias_hh_l1.grad)
