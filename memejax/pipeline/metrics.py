from collections.abc import Callable
from functools import partial
from typing import Any, cast

import jax
import jax.numpy as jnp
from flax import struct
from flax.core.scope import VariableDict
from jax import Array
from jax.typing import ArrayLike

from memejax.jnp import (
    ArrayMap,
    ModelOutput,
    jnp_geometric_mean,
    jnp_identity,
    jnp_soft_max,
    jnp_soft_min,
)

# Collapsed version of MetricCollection.
DeviceMetricDict = dict[str, ArrayLike]
LocalMetricDict = dict[str, float]


class MetricCollection:
    def compute_metrics(self) -> DeviceMetricDict:
        raise NotImplementedError

    def merge(self) -> "MetricCollection":
        """Lets us merge a vmapped MetricCollection"""
        merged = {}
        for name, metric in vars(self).items():
            if isinstance(metric, Metric):
                merged[name] = metric.merge()
            else:
                merged[name] = metric
        return type(self)(**merged)

    def traced_mdict(self) -> DeviceMetricDict:
        dmdict: DeviceMetricDict = self.compute_metrics()
        assert "loss" in dmdict, "loss required"
        return dmdict

    def local_mdict(self) -> LocalMetricDict:
        return get_local_mdict(self.traced_mdict())


def get_local_mdict(dmdict: DeviceMetricDict) -> LocalMetricDict:
    dmdict = jax.device_get(dmdict)
    mdict: LocalMetricDict = jax.tree_map(lambda x: x.item(), dmdict)
    return mdict


def pretty_print_mdict(mdict: LocalMetricDict) -> None:
    for k, v in mdict.items():
        print(f"  {k}: {v:.6f}")


MetricCollectionMap = dict[str, MetricCollection]
MetricsFn = Callable[[VariableDict, ModelOutput, ArrayMap], MetricCollectionMap]


@struct.dataclass
class Metric:
    vals: Array
    eval_fn: Callable[[Array], Array] = struct.field(pytree_node=False)

    @jax.jit
    def merge(self) -> "Metric":
        return type(self)(vals=jnp.concatenate(self.vals), eval_fn=self.eval_fn)

    @jax.jit
    def compute(self) -> Array:
        return self.eval_fn(self.vals)


@struct.dataclass
class SoftMinMetric(Metric):
    eval_fn = partial(jnp_soft_min, ep=1)

    @classmethod
    def from_values(cls, vals: Array) -> "SoftMinMetric":
        return cls(vals=jnp.array([vals]), eval_fn=cls.eval_fn)


@struct.dataclass
class SoftMaxMetric(Metric):
    eval_fn = partial(jnp_soft_max, ep=1)

    @classmethod
    def from_values(cls, vals: Array) -> "SoftMaxMetric":
        return cls(vals=jnp.array([vals]), eval_fn=cls.eval_fn)


# This is faster than the inbuilt clu metrics version, since it doesn't do
# incremental operations.
@struct.dataclass
class MeanMetric(Metric):
    eval_fn = jnp.mean

    @classmethod
    def from_values(cls, vals: Array) -> "MeanMetric":
        return cls(vals=jnp.array([vals]), eval_fn=cls.eval_fn)


@struct.dataclass
class GeometricMeanMetric(Metric):
    eval_fn = jnp_geometric_mean

    @classmethod
    def from_values(cls, vals: Array) -> "GeometricMeanMetric":
        return cls(vals=jnp.array([vals]), eval_fn=cls.eval_fn)


@struct.dataclass
class CollectionMetric(Metric):
    eval_fn = jnp_identity

    @classmethod
    def from_values(cls, vals: Array) -> "CollectionMetric":
        return cls(vals=jnp.array([vals]), eval_fn=cls.eval_fn)


@struct.dataclass
class LogitAccuracyMetric(Metric):
    eval_fn = jnp.mean

    @classmethod
    def from_values(cls, *, logits: Array, labels: Array) -> "LogitAccuracyMetric":
        return cls(vals=jnp.array([(logits.argmax(axis=-1) == labels)]), eval_fn=cls.eval_fn)


@struct.dataclass
class ClfMetrics(MetricCollection):
    """Metrics for a classifier."""

    loss: MeanMetric
    accuracy: LogitAccuracyMetric

    def compute_metrics(self) -> DeviceMetricDict:
        return {"loss": self.loss.compute(), "accuracy": self.accuracy.compute()}


@struct.dataclass
class LossMetrics(MetricCollection):
    """Metrics for loss only."""

    loss: MeanMetric

    def compute_metrics(self) -> DeviceMetricDict:
        return {"loss": self.loss.compute()}


# Be careful about jit recompilations with differing metric sizes for this.
@jax.jit
def jnp_add_dmdicts(a: DeviceMetricDict, b: DeviceMetricDict) -> DeviceMetricDict:
    if len(a) == 0:
        return b
    if len(b) == 0:
        return a
    return cast(DeviceMetricDict, jax.tree_map(jnp.add, a, b))


def slow_add_mdicts(
    a: LocalMetricDict | DeviceMetricDict, b: LocalMetricDict | DeviceMetricDict
) -> Any:
    return {k: a.get(k, 0.0) + b.get(k, 0.0) for k in set(a) | set(b)}


def slow_div_mdict(a: LocalMetricDict | DeviceMetricDict, d: float) -> Any:
    return {k: v / d for k, v in a.items()}
