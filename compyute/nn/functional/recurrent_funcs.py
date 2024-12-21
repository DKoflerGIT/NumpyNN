"""Neural network recurrent functions."""

from typing import Literal, Optional

from ...tensor_ops.creation_ops import empty, empty_like, zeros, zeros_like
from ...tensor_ops.shape_ops import concat, split
from ...tensors import ShapeError, Tensor
from .activation_funcs import ReLUFunction, SigmoidFunction, TanhFunction
from .functions import Function, FunctionContext, PseudoContext
from .linear_funcs import LinearFunction

__all__ = ["recurrent", "lstm", "gru"]


class RecurrentFunction(Function):
    """Applies the Elman recurrent function to a tensor."""

    @staticmethod
    def forward(
        ctx: FunctionContext,
        x: Tensor,
        w_i: Tensor,
        b_i: Optional[Tensor],
        w_h: Tensor,
        b_h: Optional[Tensor],
        activation: str,
    ) -> Tensor:
        if x.ndim != 3:
            raise ShapeError(f"Expected input to be 3D, got {x.ndim}D.")
        if activation not in {"relu", "tanh"}:
            raise ValueError("Activation must be either 'relu' or 'tanh'.")
        act = TanhFunction if activation == "tanh" else ReLUFunction

        # input projection W_i * x_t + b_i
        x_h = LinearFunction.forward(ctx, x, w_i, b_i)

        h = zeros_like(x_h)
        for t in range(x.shape[1]):

            # hidden projection W_h * h_t-1 + b_h
            h_h = LinearFunction.forward(ctx, h[:, t - 1], w_h, b_h)

            # apply activation h_t = act(x_t + h_h)
            h[:, t] = act.forward(ctx, x_h[:, t] + h_h)

        ctx.add(h.shape, act, b_i is not None)
        return h

    @staticmethod
    def backward(
        ctx: FunctionContext, dy: Tensor
    ) -> tuple[Tensor, Tensor, Optional[Tensor], Tensor, Optional[Tensor]]:
        h_shape, act, b = ctx.get()

        B, T, H = h_shape
        dh = zeros((B, H), device=dy.device, dtype=dy.dtype)
        dw_h = zeros((H, H), device=dy.device, dtype=dy.dtype)
        db_h = None if not b else zeros((H,), device=dy.device, dtype=dy.dtype)
        dpreact = empty((B, T, H), device=dy.device, dtype=dy.dtype)

        for t in range(T - 1, -1, -1):

            # activation gradients
            dpreact[:, t] = act.backward(ctx, dh + dy[:, t])

            # hidden projection gradients
            dh, dw_h_t, db_h_t = LinearFunction.backward(ctx, dpreact[:, t])

            # accumulate parameter gradients
            if t > 0:
                dw_h += dw_h_t
            if db_h_t:
                db_h += db_h_t

        # input projection gradients
        dx, dw_i, db_i = LinearFunction.backward(ctx, dpreact)

        return dx, dw_i, db_i, dw_h, db_h


def recurrent(
    x: Tensor,
    w_i: Tensor,
    b_i: Optional[Tensor],
    w_h: Tensor,
    b_h: Optional[Tensor],
    activation: Literal["relu", "tanh"] = "tanh",
) -> Tensor:
    """Applies the Elman recurrent function to a tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    w_i : Tensor
        Weight tensor for the input projection.
    b_i : Tensor, optional
        Bias tensor for the input projection.
    w_h : Tensor
        Weight tensor for the hidden projection.
    b_h : Tensor, optional
        Bias tensor for the hidden projection.
    activation : Literal["relu", "tanh"], optional
        Activation function to use. Defaults to ``tanh``.

    Returns
    -------
    Tensor
        Output tensor.

    See Also
    ----------
    :class:`compyute.nn.Recurrent`
    """
    return RecurrentFunction.forward(PseudoContext(), x, w_i, b_i, w_h, b_h, activation)


class LSTMFunction(Function):
    """Applies the LSTM recurrent function to a tensor."""

    @staticmethod
    def forward(
        ctx: FunctionContext,
        x: Tensor,
        w_i: Tensor,
        b_i: Optional[Tensor],
        w_h: Tensor,
        b_h: Optional[Tensor],
        activation: str,
    ) -> Tensor:
        if x.ndim != 3:
            raise ShapeError(f"Expected input to be 3D, got {x.ndim}D.")
        if activation not in {"relu", "tanh"}:
            raise ValueError("Activation must be either 'relu' or 'tanh'.")
        act = TanhFunction if activation == "tanh" else ReLUFunction

        # input projection W_i * x_t + b_i
        x_ifgo = LinearFunction.forward(ctx, x, w_i, b_i)
        x_i, x_f, x_g, x_o = split(x_ifgo, 4, dim=-1)

        i, f, g, o = [empty_like(x_i) for _ in range(4)]
        c, act_c, h = [zeros_like(x_i) for _ in range(3)]
        for t in range(x.shape[1]):

            # hidden projection W_h * h_t-1 + b_h
            h_ifgo = LinearFunction.forward(ctx, h[:, t - 1], w_h, b_h)
            h_i, h_f, h_g, h_o = split(h_ifgo, 4, dim=-1)

            # gates
            i[:, t] = SigmoidFunction.forward(ctx, x_i[:, t] + h_i)  # input gate
            f[:, t] = SigmoidFunction.forward(ctx, x_f[:, t] + h_f)  # forget gate
            o[:, t] = SigmoidFunction.forward(ctx, x_o[:, t] + h_o)  # output gate

            # candidate cell state
            g[:, t] = act.forward(ctx, x_g[:, t] + h_g)

            # cell state c_t = f_t * c_t-1 + i_t * g_t
            c[:, t] = f[:, t] * c[:, t - 1] + i[:, t] * g[:, t]

            # hidden state h_t = o_t * act(c_t)
            act_c[:, t] = act.forward(ctx, c[:, t])
            h[:, t] = o[:, t] * act_c[:, t]

        ctx.add(i, f, g, o, b_i is not None, c, act, act_c)
        return h

    @staticmethod
    def backward(
        ctx: FunctionContext, dy: Tensor
    ) -> tuple[Tensor, Tensor, Optional[Tensor], Tensor, Optional[Tensor]]:
        i, f, g, o, b, c, act, act_c = ctx.get()

        B, T, H = i.shape
        dh = zeros((B, H), device=dy.device, dtype=dy.dtype)
        dw_h = zeros((4 * H, H), device=dy.device, dtype=dy.dtype)
        db_h = None if not b else zeros((4 * H,), device=dy.device, dtype=dy.dtype)
        difgo_preact = empty((B, T, 4 * H), device=dy.device, dtype=dy.dtype)
        dc = zeros_like(c)

        for t in range(T - 1, -1, -1):
            # hidden state gradients
            dh += dy[:, t]
            do = act_c[:, t] * dh
            dc[:, t] += act.backward(ctx, dh) * o[:, t]

            # cell state gradients
            df = zeros_like(dh) if t < 1 else c[:, t - 1] * dc[:, t]
            if t > 0:
                dc[:, t - 1] += f[:, t] * dc[:, t]
            di = g[:, t] * dc[:, t]
            dg = i[:, t] * dc[:, t]

            # candidate cell state gradients
            dg_hat = act.backward(ctx, dg)

            # gate gradients
            do_preact = SigmoidFunction.backward(ctx, do)
            df_preact = SigmoidFunction.backward(ctx, df)
            di_preact = SigmoidFunction.backward(ctx, di)

            # hidden projection gradients
            difgo_preact[:, t] = concat([di_preact, df_preact, dg_hat, do_preact])
            dh, dw_h_t, db_h_t = LinearFunction.backward(ctx, difgo_preact[:, t])

            # accumulate parameter gradients
            if t > 0:
                dw_h += dw_h_t
            if db_h_t:
                db_h += db_h_t

        # input projection gradients
        dx, dw_i, db_i = LinearFunction.backward(ctx, difgo_preact)

        return dx, dw_i, db_i, dw_h, db_h


def lstm(
    x: Tensor,
    w_i: Tensor,
    b_i: Optional[Tensor],
    w_h: Tensor,
    b_h: Optional[Tensor],
    activation: Literal["relu", "tanh"] = "tanh",
) -> Tensor:
    """Applies the LSTM recurrent function to a tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    w_i : Tensor
        Weight tensor for the input projection.
    b_i : Tensor, optional
        Bias tensor for the input projection.
    w_h : Tensor
        Weight tensor for the hidden projection.
    b_h : Tensor, optional
        Bias tensor for the hidden projection.
    activation : Literal["relu", "tanh"], optional
        Activation function to use. Defaults to ``tanh``.

    Returns
    -------
    Tensor
        Output tensor.

    See Also
    ----------
    :class:`compyute.nn.LSTM`
    """
    return LSTMFunction.forward(PseudoContext(), x, w_i, b_i, w_h, b_h, activation)


class GRUFunction(Function):
    """Applies the GRU recurrent function to a tensor."""

    @staticmethod
    def forward(
        ctx: FunctionContext,
        x: Tensor,
        w_i: Tensor,
        b_i: Optional[Tensor],
        w_h: Tensor,
        b_h: Optional[Tensor],
        activation: str,
    ) -> Tensor:
        if x.ndim != 3:
            raise ShapeError(f"Expected input to be 3D, got {x.ndim}D.")
        if activation not in {"relu", "tanh"}:
            raise ValueError("Activation must be either 'relu' or 'tanh'.")
        act = TanhFunction if activation == "tanh" else ReLUFunction

        # input projection W_i * x_t + b_i
        x_rzn = LinearFunction.forward(ctx, x, w_i, b_i)
        x_r, x_z, x_n = split(x_rzn, 3, dim=-1)

        r, z, n = [empty_like(x_r) for _ in range(3)]
        h_n, h = [zeros_like(x_r) for _ in range(2)]
        for t in range(x.shape[1]):

            # hidden projection W_h * h_t-1 + b_h
            h_rzn = LinearFunction.forward(ctx, h[:, t - 1], w_h, b_h)
            h_r, h_z, h_n[:, t] = split(h_rzn, 3, dim=-1)

            # gates
            r[:, t] = SigmoidFunction.forward(ctx, x_r[:, t] + h_r)  # reset gate
            z[:, t] = SigmoidFunction.forward(ctx, x_z[:, t] + h_z)  # update gate

            # candidate hidden state n_t = act(x_n + r_t * h_t-1)
            n[:, t] = act.forward(ctx, x_n[:, t] + r[:, t] * h_n[:, t])

            # hidden state h_t = (1 - z_t) * n_t + z_t * h_t-1
            h[:, t] = (1 - z[:, t]) * n[:, t] + z[:, t] * h[:, t - 1]

        ctx.add(r, z, n, b_i is not None, h_n, act, h)
        return h

    @staticmethod
    def backward(
        ctx: FunctionContext, dy: Tensor
    ) -> tuple[Tensor, Tensor, Optional[Tensor], Tensor, Optional[Tensor]]:
        r, z, n, b, h_n, act, h = ctx.get()

        B, T, H = r.shape
        dh = zeros((B, H), device=dy.device, dtype=dy.dtype)
        dw_h = zeros((3 * H, H), device=dy.device, dtype=dy.dtype)
        db_h = None if not b else zeros((3 * H,), device=dy.device, dtype=dy.dtype)
        drzn_preact = empty((B, T, 3 * H), device=dy.device, dtype=dy.dtype)

        for t in range(T - 1, -1, -1):
            # hidden state gradients
            dh += dy[:, t]
            dz = ((0 if t < 1 else h[:, t - 1]) - n[:, t]) * dh
            dn = (1 - z[:, t]) * dh
            dh = z[:, t] * dh

            # candidate hidden state gradients
            dn_preact = act.backward(ctx, dn)
            dr = h_n[:, t] * dn_preact

            # gate gradients
            dz_preact = SigmoidFunction.backward(ctx, dz)
            dr_preact = SigmoidFunction.backward(ctx, dr)

            # hidden projection gradients
            drzn_preact[:, t] = concat([dr_preact, dz_preact, dn_preact])
            drzn_preact_h = concat([dr_preact, dz_preact, r[:, t] * dn_preact])
            dh_t, dw_h_t, db_h_t = LinearFunction.backward(ctx, drzn_preact_h)
            dh += dh_t

            # accumulate parameter gradients
            if t > 0:
                dw_h += dw_h_t
            if db_h_t:
                db_h += db_h_t

        # input projection gradients
        dx, dw_i, db_i = LinearFunction.backward(ctx, drzn_preact)

        return dx, dw_i, db_i, dw_h, db_h


def gru(
    x: Tensor,
    w_i: Tensor,
    b_i: Optional[Tensor],
    w_h: Tensor,
    b_h: Optional[Tensor],
    activation: Literal["relu", "tanh"] = "tanh",
) -> Tensor:
    """Applies the GRU recurrent function to a tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    w_i : Tensor
        Weight tensor for the input projection.
    b_i : Tensor, optional
        Bias tensor for the input projection.
    w_h : Tensor
        Weight tensor for the hidden projection.
    b_h : Tensor, optional
        Bias tensor for the hidden projection.
    activation : Literal["relu", "tanh"], optional
        Activation function to use. Defaults to ``tanh``.

    Returns
    -------
    Tensor
        Output tensor.

    See Also
    ----------
    :class:`compyute.nn.GRU`
    """
    return GRUFunction.forward(PseudoContext(), x, w_i, b_i, w_h, b_h, activation)
