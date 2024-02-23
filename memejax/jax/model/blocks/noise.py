import copy
import dataclasses
from dataclasses import dataclass

import flax.linen as nn
import jax
import optuna
import typing_extensions
from dataclasses_json import DataClassJsonMixin, Undefined, dataclass_json
from jax import Array

from memejax.jax.hyperparam.trial import OptunaParameterable, OptunaSearchCfg
from memejax.jax.util import ArrayOrMap, apply_arrayormap


@dataclass_json(undefined=Undefined.RAISE)
@dataclass(eq=True, kw_only=True, order=True, frozen=True)
class GaussianNoiseBlkCfg(OptunaParameterable, DataClassJsonMixin):
    mean: float = 0.0
    stddev: float = 1.0

    @typing_extensions.override
    def optuna_params(
        self, trial: optuna.Trial, optuna_cfg: OptunaSearchCfg, prefix: str = ""
    ) -> "GaussianNoiseBlkCfg":
        cfg = copy.deepcopy(self)
        if optuna_cfg.model_params:
            cfg = dataclasses.replace(
                cfg, stddev=trial.suggest_float(prefix + "stddev", 0.0, 1.0, step=0.001)
            )
        return cfg


class GaussianNoiseBlk(nn.Module):
    blk_cfg: GaussianNoiseBlkCfg

    def apply_array(self, _key: str, x: Array, train: bool) -> Array:
        if train:
            rng = self.make_rng("dropout")  # use dropout sequence for convenience
            noise = jax.random.normal(key=rng, shape=x.shape)
            x = x + noise * self.blk_cfg.stddev + self.blk_cfg.mean
        return x

    @nn.compact
    def __call__(self, inp: ArrayOrMap, train: bool) -> ArrayOrMap:
        return apply_arrayormap(inp, train, self.apply_array)
