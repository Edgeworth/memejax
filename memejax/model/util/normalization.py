from typing import cast

import numpy as np
import numpy.typing as npt

from memejax.jnp import np_pct_diff


def np_normalize_pct_diff(data: npt.NDArray) -> npt.NDArray:
    return np.insert(np_pct_diff(data), 0, 1.0, axis=-1)


def np_normalize_log_pct_diff(data: npt.NDArray) -> npt.NDArray:
    return cast(npt.NDArray, np.log(np_normalize_pct_diff(data) + 1.0))


def np_normalize_minmax(data: npt.NDArray) -> npt.NDArray:
    minimum = np.min(data, axis=-1, keepdims=True)
    maximum = np.max(data, axis=-1, keepdims=True)
    return cast(npt.NDArray, (data - minimum) / (maximum - minimum + 1e-8))


def np_normalize_mean(data: npt.NDArray) -> npt.NDArray:
    minimum = np.min(data, axis=-1, keepdims=True)
    maximum = np.max(data, axis=-1, keepdims=True)
    mean = np.mean(data, axis=-1, keepdims=True)
    return cast(npt.NDArray, (data - mean) / (maximum - minimum + 1e-8))


def np_normalize_zscore(data: npt.NDArray) -> npt.NDArray:
    mean = np.mean(data, axis=-1, keepdims=True)
    std = np.std(data, axis=-1, keepdims=True)
    return cast(npt.NDArray, (data - mean) / (std + 1e-8))
