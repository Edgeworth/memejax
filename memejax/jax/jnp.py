from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from chex import assert_rank
from jax import Array
from jax.typing import ArrayLike


def jnp_pad_left_edge(arr: Array, size: int) -> Array:
    """Pad an array on the left side with the edge value."""
    if len(arr) == 0:
        return jnp.zeros(size)
    pad_width = max(size - len(arr), 0)
    if pad_width == 0:
        return arr
    return jnp.pad(arr, (pad_width, 0), mode="edge")


def jnp_pad_left_value(arr: Array, size: int, value: float) -> Array:
    """Pad an array on the left side with the edge value."""
    pad_width = max(size - len(arr), 0)
    if pad_width == 0:
        return arr
    return jnp.pad(arr, (pad_width, 0), mode="constant", constant_values=value)


@jax.jit
def jnp_pct_diff(arr: Array) -> Array:
    """Compute percentage difference with a nice gradient and avoiding division by zero."""
    diffs = jnp.diff(arr)
    divisor = arr[..., :-1]
    divisor = jnp.where(divisor == 0.0, divisor + 1e-8, divisor)
    return cast(Array, diffs / divisor)


def np_pct_diff(arr: npt.NDArray) -> npt.NDArray:
    """Compute percentage difference on the last axis of the data."""
    diffs = np.diff(arr, axis=-1)
    divisor = arr[..., :-1]
    divisor = np.where(divisor == 0.0, divisor + 1e-8, divisor)
    return np.divide(diffs, divisor)


@jax.jit
def jnp_soft_max(arr: Array, ep: float = 1e2) -> Array:
    """Computes differentiable maximum value of an array"""
    # Scale to make value closer to max.
    prop_ep = jnp.abs(jnp.sum(arr)) * ep
    return cast(Array, jnp.sum(jax.nn.softmax(arr * prop_ep) * arr, axis=-1))


@jax.jit
def jnp_soft_min(arr: Array, ep: float = 1e2) -> Array:
    """Computes differentiable minimum value of an array"""
    # Scale to make value closer to min.
    prop_ep = jnp.abs(jnp.sum(arr)) * ep
    return cast(Array, jnp.sum(jax.nn.softmax(-arr * prop_ep) * arr, axis=-1))


def jnp_upper_tri(size: int, include_diag: bool = True) -> Array:
    """Computes an upper triangular matrix of ones."""
    return cast(Array, jnp.triu(jnp.ones((size, size)), k=0 if include_diag else 1))


def jnp_lower_tri(size: int, include_diag: bool = True) -> Array:
    """Computes an upper triangular matrix of ones."""
    return cast(Array, jnp.tril(jnp.ones((size, size)), k=0 if include_diag else -1))


@jax.jit
def jnp_cum_soft_max(arr: Array) -> Array:
    """Computes differentiable cumulative max of an array"""
    tril = jnp_lower_tri(len(arr))
    cums = tril * arr - 1e8 * (1 - tril)
    return cast(Array, jnp_soft_max(cums))


@jax.jit
def jnp_cum_soft_min(arr: Array) -> Array:
    """Computes differentiable cumulative min of an array"""
    tril = jnp_lower_tri(len(arr))
    cums = tril * arr + 1e8 * (1 - tril)
    return cast(Array, jnp_soft_min(cums))


@jax.jit
def jnp_leaky_heaviside(
    x: Array, threshold: float, transition_width: float = 1e-6, out_slope: float = 0.01
) -> Array:
    # Keep transition width small, as it isn't that important but can cause
    # issues if it's too big.
    x = x - threshold
    transition = transition_width / 2.0
    left = jnp.where(x <= -transition, out_slope * (x + transition), 0.0)
    mid = jnp.where(jnp.abs(x) <= transition, 0.5 * x / transition + 0.5, 0.0)
    right = jnp.where(x >= transition, out_slope * (x - transition) + 1.0, 0.0)
    return left + mid + right


@jax.jit
def jnp_geometric_mean(arr: Array) -> Array:
    """Computes differentiable geometric mean of an array"""
    return jnp.exp(jnp.mean(jnp.log(arr)))


@jax.jit
def jnp_sharpe(returns: jax.Array, riskfreerate: jax.Array) -> Array:
    assert_rank(returns, 1)
    # Add small value to variance to ensure sqrt is differentiable. This has
    # the bonus of not requiring a select which will introduce a
    # non-continous derivative.
    stddev = jnp_stddev(returns)
    sharpe = (returns.mean() - riskfreerate) / stddev
    return cast(Array, sharpe)


@jax.jit
def jnp_sharpe_norfr(returns: jax.Array) -> Array:
    return cast(Array, jnp_sharpe(returns, 0.0))


@jax.jit
def jnp_sortino(returns: jax.Array, riskfreerate: jax.Array) -> Array:
    assert_rank(returns, 1)
    # Compute downside deviation, which is just setting all the positive values to 0.
    neg_returns = jnp.where(returns < 0.0, returns, 0.0)
    stddev = jnp_stddev(neg_returns)
    # Multiply sortino ratio by sqrt(2) to match the sharpe ratio if
    # downside and upside gains are equally volatile.
    sortino = (returns.mean() - riskfreerate) / (stddev * jnp.sqrt(2.0))
    return cast(Array, sortino)


@jax.jit
def jnp_sortino_norfr(returns: jax.Array) -> Array:
    return cast(Array, jnp_sortino(returns, 0.0))


@jax.jit
def jnp_max_drawdown(log_returns: jax.Array) -> Array:
    # Compute surrogate values for using to compute max drawdown.
    # We start with a value of one then apply the returns to it.
    values = jnp.concatenate([jnp.ones(1), jnp.exp(jnp.cumsum(log_returns))])
    assert_rank(values, 1)
    # Compute the backward cumulative min.
    cummin = jnp.flip(jnp_cum_soft_min(jnp.flip(values)))
    # Compute the drawdown for each value as a percentage of the value
    drawdown = (values - cummin) / (values + 1e-8)
    return cast(Array, jnp_soft_max(drawdown))


@jax.jit
def jnp_identity(x: jax.Array) -> Array:
    return x


@jax.jit
def jnp_stddev(x: jax.Array) -> Array:
    """Safe and differentiable stddev."""
    return jnp.sqrt(jnp.var(x) + 1e-8)


def hvp(f: Callable, primals: tuple, tangents: tuple) -> Array:
    """Hessian-vector product."""
    hvp = jax.jvp(jax.grad(f), primals, tangents)[1]
    return cast(Array, hvp)


def want_below_lin(x: Array, *, thresh: ArrayLike, hate_above: float, love_below: float) -> Array:
    x = x - thresh
    return jnp.where(x >= 0, hate_above * x, love_below * x)


def want_above_lin(x: Array, *, thresh: ArrayLike, hate_below: float, love_above: float) -> Array:
    x = x - thresh
    return jnp.where(x >= 0, -love_above * x, -hate_below * x)


def want_below_exp(x: Array, *, thresh: ArrayLike, hate_above: float, love_below: float) -> Array:
    x = x - thresh
    return jnp.where(x >= 0, hate_above**x, -(love_below ** (-x)))


def want_above_exp(x: Array, *, thresh: ArrayLike, hate_below: float, love_above: float) -> Array:
    x = x - thresh
    return jnp.where(x >= 0, -(love_above**x), hate_below ** (-x))
