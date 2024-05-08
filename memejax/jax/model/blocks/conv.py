import copy
import dataclasses
from dataclasses import dataclass
from typing import override

import flax.linen as nn
import jax
import jax.numpy as jnp
import optuna
from chex import assert_shape
from dataclasses_json import DataClassJsonMixin, Undefined, dataclass_json
from jax import Array

from memejax.jax.hyperparam.trial import OptunaParameterable, OptunaSearchCfg
from memejax.jax.model.blocks.cfg import ModelCfg, Variable
from memejax.jax.model.layers.mlp import MlpLayer
from memejax.jax.util import JaxArrayMap


@dataclass_json(undefined=Undefined.RAISE)
@dataclass(eq=True, kw_only=True, order=True, frozen=True)
class ConvBlkCfg(OptunaParameterable, DataClassJsonMixin):
    conv_layers: int = 4
    conv_features: int = 16
    dense_layers: int = 1
    dense_features: int = 1024
    outputs: Variable

    @override
    def optuna_params(
        self, trial: optuna.Trial, optuna_cfg: OptunaSearchCfg, prefix: str = ""
    ) -> "ConvBlkCfg":
        cfg = copy.deepcopy(self)
        if optuna_cfg.model_dims:
            cfg = dataclasses.replace(
                cfg,
                conv_layers=trial.suggest_int(prefix + "conv_layers", 1, self.conv_layers),
                conv_features=trial.suggest_int(
                    prefix + "conv_features", 1, self.conv_features, log=True
                ),
                dense_layers=trial.suggest_int(prefix + "dense_layers", 0, self.dense_layers),
                dense_features=trial.suggest_int(
                    prefix + "dense_features", 1, self.dense_features, log=True
                ),
            )
        return cfg


class ConvBlk(nn.Module):
    cfg: ModelCfg
    blk_cfg: ConvBlkCfg

    @nn.compact
    def __call__(self, data: JaxArrayMap, train: bool) -> Array:
        cfg = self.cfg
        blk_cfg = self.blk_cfg

        x = data["x"]
        x = nn.Dropout(rate=cfg.input_dropout, deterministic=not train)(x)
        inp = x

        # conv needs these dimensions: (batch, height, width, channels), so expand dimensions:
        x = x[jnp.newaxis, ..., jnp.newaxis]

        for _ in range(blk_cfg.conv_layers):
            x = nn.Conv(features=blk_cfg.conv_features, kernel_size=(3, 3))(x)
            x = nn.leaky_relu(x)
            x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
            x = nn.Dropout(rate=cfg.dropout, deterministic=not train)(x)

        x = x.reshape(-1)
        # Mix input in as well
        x = jnp.concatenate([x, inp], axis=0)

        for _ in range(blk_cfg.dense_layers):
            x = MlpLayer.from_cfg(cfg=cfg, features=blk_cfg.dense_features)(x, train)

        outputs = cfg.resolve_int(blk_cfg.outputs)
        x = nn.Dense(outputs)(x)

        x = jax.nn.softmax(x)  # Map alloc so its sum is 1.
        assert_shape(x, (outputs,))  # output should be a vector of length outputs
        return x
