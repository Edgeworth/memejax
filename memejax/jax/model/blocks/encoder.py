import copy
import dataclasses
from dataclasses import dataclass

import flax.linen as nn
import jax
import numpy as np
import optuna
import typing_extensions
from chex import assert_shape
from dataclasses_json import DataClassJsonMixin, Undefined, dataclass_json
from jax import Array

from memejax.jax.hyperparam.trial import OptunaParameterable, OptunaSearchCfg
from memejax.jax.model.blocks.cfg import ModelCfg, Variable
from memejax.jax.model.layers.mlp import MlpLayer
from memejax.jax.util import JaxArrayMap


@dataclass_json(undefined=Undefined.RAISE)
@dataclass(eq=True, kw_only=True, order=True, frozen=True)
class EncoderBlkCfg(OptunaParameterable, DataClassJsonMixin):
    num_layers: int = 2
    bottleneck: int = 4
    features: int = 32
    outputs: Variable

    @typing_extensions.override
    def optuna_params(
        self, trial: optuna.Trial, optuna_cfg: OptunaSearchCfg, prefix: str = ""
    ) -> "EncoderBlkCfg":
        cfg = copy.deepcopy(self)
        if optuna_cfg.model_dims:
            cfg = dataclasses.replace(
                cfg,
                num_layers=trial.suggest_int(prefix + "num_layers", 1, self.num_layers),
                bottleneck=trial.suggest_int(
                    prefix + "bottleneck_features", 1, self.bottleneck, log=True
                ),
                features=trial.suggest_int(prefix + "dense_features", 1, self.features, log=True),
            )
        return cfg


class EncoderBlk(nn.Module):
    cfg: ModelCfg
    blk_cfg: EncoderBlkCfg

    @nn.compact
    def __call__(self, data: JaxArrayMap, train: bool) -> Array:
        cfg = self.cfg
        blk_cfg = self.blk_cfg

        x = data["x"]
        x = nn.Dropout(rate=cfg.input_dropout, deterministic=not train)(x)

        features = np.geomspace(
            blk_cfg.features, blk_cfg.bottleneck, num=blk_cfg.num_layers, dtype=int
        )
        for num in features:
            x = MlpLayer.from_cfg(cfg=cfg, features=num)(x, train)

        x = MlpLayer(dropout=0.0, features=blk_cfg.bottleneck, layer_norm=cfg.layer_norm)(x, train)

        for num in reversed(features):
            x = MlpLayer.from_cfg(cfg=cfg, features=num)(x, train)

        outputs = cfg.resolve_int(blk_cfg.outputs)
        x = nn.Dense(outputs)(x)
        x = jax.nn.softmax(x)  # Map alloc so its sum is 1.
        assert_shape(x, (outputs,))  # output should be a vector of length outputs

        return x
