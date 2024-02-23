from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array


def initialize_identity(_: Array, shape: Sequence[int], dtype: type = jnp.float_) -> Array:
    # All shape dimensions must be the same for identity
    assert all(shape[0] == s for s in shape)
    return jnp.identity(shape[0], jax.dtypes.canonicalize_dtype(dtype))


def sinusoidal_emb(
    seq_len: int, d_emb: int, min_scale: float = 1.0, max_scale: float = 10000.0
) -> Array:
    """Sinusoidal positional encoding.

    Args:
        min_scale: minimum frequency scale in sinusoid
        max_scale: maximum frequency scale in sinusoid

    Returns:
        Array of (seq_len, d_emb)
    """
    # Compute max_scale / min_scale ^ (2i / d_emb) to use as input for sin and cos.
    scale_factor = -np.log(max_scale / min_scale) / d_emb
    div = min_scale * np.exp(np.arange(0, d_emb, 2) * scale_factor)

    # Compute position index (index of value in input sequence).
    pos = np.arange(seq_len)[..., np.newaxis]

    pos_emb = np.zeros((seq_len, d_emb))
    pos_emb[:, 0::2] = np.sin(pos * div)
    pos_emb[:, 1::2] = np.cos(pos * div)
    return jnp.asarray(pos_emb)


def initialize_sinusoidal(
    min_scale: float = 1.0, max_scale: float = 10000.0
) -> Callable[[Array, Sequence[int], type], Array]:
    def fn(_key: Array, shape: Sequence[int], _dtype: type = jnp.float_) -> Array:
        # Expect (seq_len, d_emb)
        assert len(shape) == 2
        seq_len, d_emb = shape
        return sinusoidal_emb(seq_len, d_emb, min_scale, max_scale)

    return fn
