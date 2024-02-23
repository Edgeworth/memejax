import copy
import dataclasses
from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp
import optuna
import typing_extensions
from chex import assert_shape
from dataclasses_json import DataClassJsonMixin, Undefined, dataclass_json
from jax import Array

from memejax.jax.hyperparam.trial import OptunaParameterable, OptunaSearchCfg, suggest_int_exp
from memejax.jax.jnp import ArrayMap
from memejax.jax.model.blocks.cfg import ModelCfg, Variable
from memejax.jax.model.blocks.posemb import PosEmbBlk, PosEmbBlkCfg
from memejax.jax.model.layers.mlp import MlpLayer


@dataclass_json(undefined=Undefined.RAISE)
@dataclass(eq=True, kw_only=True, order=True, frozen=True)
class TransformerBlkCfg(OptunaParameterable, DataClassJsonMixin):
    d_emb: int = 16
    num_heads: int = 1
    num_layers: int = 1
    attn_dropout: float = 0.1
    outputs: Variable

    @typing_extensions.override
    def optuna_params(
        self, trial: optuna.Trial, optuna_cfg: OptunaSearchCfg, prefix: str = ""
    ) -> "TransformerBlkCfg":
        cfg = copy.deepcopy(self)
        if optuna_cfg.model_dims:
            cfg = dataclasses.replace(
                cfg,
                # Need to ensure that d_emb is divisible by num_heads.
                d_emb=suggest_int_exp(trial, prefix + "d_emb", 1, self.d_emb, base=2),
                num_heads=suggest_int_exp(trial, prefix + "num_heads", 0, self.num_heads, base=2),
                num_layers=trial.suggest_int(prefix + "num_layers", 1, self.num_layers),
                attn_dropout=trial.suggest_float(prefix + "attn_dropout", 0.0, 1.0, step=0.05),
            )
        return cfg


class TransformerEncoderBlk(nn.Module):
    cfg: ModelCfg
    blk_cfg: TransformerBlkCfg

    @nn.compact
    def __call__(self, x: Array, train: bool) -> Array:
        cfg = self.cfg
        blk_cfg = self.blk_cfg
        assert_shape(x, (None, blk_cfg.d_emb))

        residual = x
        x = nn.SelfAttention(num_heads=blk_cfg.num_heads, dropout_rate=blk_cfg.attn_dropout)(
            x, deterministic=not train
        )

        x = nn.Dropout(rate=cfg.dropout, deterministic=not train)(x)
        x = nn.LayerNorm()(x + residual)
        residual = x

        x = nn.Dense(features=blk_cfg.d_emb)(x)
        x = nn.Dropout(rate=cfg.dropout, deterministic=not train)(x)
        x = nn.LayerNorm()(x + residual)

        return x


class TransformerBlk(nn.Module):
    cfg: ModelCfg
    blk_cfg: TransformerBlkCfg

    @nn.compact
    def __call__(self, data: ArrayMap, train: bool) -> Array:
        cfg = self.cfg
        blk_cfg = self.blk_cfg

        x = data["x"]
        x = nn.Dropout(rate=cfg.input_dropout, deterministic=not train)(x)
        x = nn.Dense(1024)(x)

        # Apply positional encoding by starting with (num_series, num_timesteps,
        # d_emb) having all the d_emb values as the same value. TODO(2): work
        # out a better way to do this.
        x = jnp.zeros([*x.shape, blk_cfg.d_emb]) + x[..., jnp.newaxis]
        # x = jax.vmap(lambda x: PositionalEmbedding(learnable=False)(x, train))(x)
        x = PosEmbBlk(cfg=cfg, blk_cfg=PosEmbBlkCfg(learnable=False))(x, train)

        # Now apply transformer layers.
        for _ in range(blk_cfg.num_layers):
            x = TransformerEncoderBlk(cfg=cfg, blk_cfg=blk_cfg)(x, train)

        x = MlpLayer.from_cfg(cfg=cfg, features=1)(x, train)
        x = x.reshape(-1)

        outputs = cfg.resolve_int(blk_cfg.outputs)
        x = nn.Dense(outputs)(x)
        x = jax.nn.softmax(x)  # Map alloc so its sum is 1.
        assert_shape(x, (outputs,))  # output should be a vector of length outputs
        return x
