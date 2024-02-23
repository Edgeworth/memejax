import copy
import dataclasses
from dataclasses import dataclass

import flax.linen as nn
import optuna
import typing_extensions
from chex import assert_shape
from dataclasses_json import DataClassJsonMixin, Undefined, dataclass_json
from jax import Array

from memejax.common import ArrayOrMap, apply_arrayormap
from memejax.hyperparam.trial import OptunaParameterable, OptunaSearchCfg
from memejax.model.blocks.cfg import ModelCfg, Variable
from memejax.model.layers.mlp import MlpLayer


@dataclass_json(undefined=Undefined.RAISE)
@dataclass(eq=True, kw_only=True, order=True, frozen=True)
class MlpBlkCfg(OptunaParameterable, DataClassJsonMixin):
    num_layers: int
    dense_features: int
    outputs: Variable

    @typing_extensions.override
    def optuna_params(
        self, trial: optuna.Trial, optuna_cfg: OptunaSearchCfg, prefix: str = ""
    ) -> "MlpBlkCfg":
        cfg = copy.deepcopy(self)
        if optuna_cfg.model_dims:
            cfg = dataclasses.replace(
                cfg,
                num_layers=trial.suggest_int(prefix + "num_layers", 1, self.num_layers),
                dense_features=trial.suggest_int(
                    prefix + "dense_features", 2, self.dense_features, log=True
                ),
            )
        return cfg


class MlpBlk(nn.Module):
    cfg: ModelCfg
    blk_cfg: MlpBlkCfg

    def apply_array(self, _key: str, x: Array, train: bool) -> Array:
        cfg = self.cfg
        blk_cfg = self.blk_cfg

        dropout = cfg.input_dropout
        for _ in range(blk_cfg.num_layers):
            x = MlpLayer(
                dropout=dropout, features=blk_cfg.dense_features, layer_norm=cfg.layer_norm
            )(x, train)
            dropout = cfg.dropout

        outputs = cfg.resolve_int(blk_cfg.outputs)
        x = nn.Dense(outputs)(x)
        assert_shape(x, (outputs,))  # output should be a vector of length outputs

        return x

    @nn.compact
    def __call__(self, inp: ArrayOrMap, train: bool) -> ArrayOrMap:
        return apply_arrayormap(inp, train, self.apply_array)
