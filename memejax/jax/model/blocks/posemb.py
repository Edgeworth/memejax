import copy
import dataclasses
from dataclasses import dataclass

import flax.linen as nn
import optuna
import typing_extensions
from chex import assert_rank
from dataclasses_json import DataClassJsonMixin, Undefined, dataclass_json
from jax import Array

from memejax.jax.hyperparam.trial import OptunaParameterable, OptunaSearchCfg
from memejax.jax.model.blocks.cfg import ModelCfg
from memejax.jax.model.util.initializers import initialize_sinusoidal, sinusoidal_emb
from memejax.jax.util import JaxArrayOrMap, apply_arrayormap


@dataclass_json(undefined=Undefined.RAISE)
@dataclass(eq=True, kw_only=True, order=True, frozen=True)
class PosEmbBlkCfg(OptunaParameterable, DataClassJsonMixin):
    learnable: bool = False

    @typing_extensions.override
    def optuna_params(
        self, trial: optuna.Trial, optuna_cfg: OptunaSearchCfg, prefix: str = ""
    ) -> "PosEmbBlkCfg":
        cfg = copy.deepcopy(self)
        if optuna_cfg.model_params:
            cfg = dataclasses.replace(
                cfg, learnable=trial.suggest_categorical(prefix + "learnable", [True, False])
            )
        return cfg


class PosEmbBlk(nn.Module):
    cfg: ModelCfg
    blk_cfg: PosEmbBlkCfg

    def apply_array(self, key: str, x: Array, train: bool) -> Array:
        assert_rank(x, 2)

        if self.blk_cfg.learnable:
            # Use a learnable positional embedding, initialized from sinusoid.
            pos_emb = self.param(f"pos_emb_{key}", initialize_sinusoidal(), x.shape)
        else:
            # Use a fixed positional embedding.
            pos_emb = sinusoidal_emb(x.shape[0], x.shape[1])

        x = x + pos_emb
        # TODO(1): dropout maybe not good for positional embedding?
        x = nn.Dropout(rate=self.cfg.dropout, deterministic=not train)(x)
        return x

    @nn.compact
    def __call__(self, inp: JaxArrayOrMap, train: bool) -> JaxArrayOrMap:
        return apply_arrayormap(inp, train, self.apply_array)
