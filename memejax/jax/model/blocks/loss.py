from dataclasses import dataclass

import flax.linen as nn
import optax
from dataclasses_json import DataClassJsonMixin, Undefined, dataclass_json

from memejax.jax.hyperparam.trial import OptunaParameterable
from memejax.jax.jnp import ArrayMap
from memejax.jax.pipeline.metrics import (
    ClfMetrics,
    LogitAccuracyMetric,
    LossMetrics,
    MeanMetric,
    MetricCollection,
)


@dataclass_json(undefined=Undefined.RAISE)
@dataclass(eq=True, kw_only=True, order=True, frozen=True)
class LossPredictionCfg(OptunaParameterable, DataClassJsonMixin):
    prediction: str = "x"
    target: str = "y"
    multiplier: float = 1.0


class LossCrossEntropyIntegerBlk(nn.Module):
    blk_cfg: LossPredictionCfg

    @nn.compact
    def __call__(self, inp: ArrayMap) -> ClfMetrics:
        x, y = inp[self.blk_cfg.prediction], inp[self.blk_cfg.target]
        cross_entropy = optax.softmax_cross_entropy_with_integer_labels(x, y).mean()
        loss = cross_entropy * self.blk_cfg.multiplier
        mcol = ClfMetrics(
            accuracy=LogitAccuracyMetric.from_values(logits=x, labels=y),
            loss=MeanMetric.from_values(loss),
        )
        return mcol


class LossMSEBlk(nn.Module):
    blk_cfg: LossPredictionCfg

    @nn.compact
    def __call__(self, inp: ArrayMap) -> MetricCollection:
        x = inp[self.blk_cfg.prediction].reshape(-1)
        y = inp.get(self.blk_cfg.target, None)
        if y is not None:
            y = y.reshape(-1)
        loss = optax.squared_error(x, targets=y).mean() * self.blk_cfg.multiplier
        mcol = LossMetrics(loss=MeanMetric.from_values(loss))
        return mcol


@dataclass_json(undefined=Undefined.RAISE)
@dataclass(eq=True, kw_only=True, order=True, frozen=True)
class LossLogPredictionCfg(OptunaParameterable, DataClassJsonMixin):
    log_prediction: str = "x"
    target: str = "y"
    multiplier: float = 1.0


class LossKLDivergenceBlk(nn.Module):
    blk_cfg: LossLogPredictionCfg

    @nn.compact
    def __call__(self, inp: ArrayMap) -> MetricCollection:
        x = inp[self.blk_cfg.log_prediction].reshape(-1)
        y = inp[self.blk_cfg.target].reshape(-1)
        loss = optax.kl_divergence(x, targets=y).mean() * self.blk_cfg.multiplier
        mcol = LossMetrics(loss=MeanMetric.from_values(loss))
        return mcol
