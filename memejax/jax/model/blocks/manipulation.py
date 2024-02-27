from dataclasses import dataclass

import flax.linen as nn
import jax.numpy as jnp
import optuna
import typing_extensions
from dataclasses_json import DataClassJsonMixin, Undefined, dataclass_json
from jax import Array

from memejax.jax.hyperparam.trial import OptunaParameterable, OptunaSearchCfg
from memejax.jax.model.blocks.cfg import EmptyBlkCfg
from memejax.jax.util import JaxArrayOrMap, apply_arrayormap


class SoftmaxBlk(nn.Module):
    def apply_array(self, _key: str, x: Array, _train: bool) -> Array:
        x = nn.softmax(x)
        return x

    @nn.compact
    def __call__(self, inp: JaxArrayOrMap, train: bool) -> JaxArrayOrMap:
        return apply_arrayormap(inp, train, self.apply_array)


@dataclass_json(undefined=Undefined.RAISE)
@dataclass(eq=True, kw_only=True, order=True, frozen=True)
class ConcatBlkCfg(OptunaParameterable, DataClassJsonMixin):
    axis: int = -1
    value: float = 0.0
    times: int = 1

    @typing_extensions.override
    def optuna_params(
        self, trial: optuna.Trial, optuna_cfg: OptunaSearchCfg, prefix: str = ""
    ) -> "ConcatBlkCfg":
        return self


class ConcatBlk(nn.Module):
    blk_cfg: ConcatBlkCfg

    def apply_array(self, _key: str, x: Array, _train: bool) -> Array:
        concat = jnp.ones_like(x) * self.blk_cfg.value
        x = jnp.concatenate([x] + [concat] * self.blk_cfg.times, axis=self.blk_cfg.axis)
        return x

    @nn.compact
    def __call__(self, inp: JaxArrayOrMap, train: bool) -> JaxArrayOrMap:
        return apply_arrayormap(inp, train, self.apply_array)


@dataclass_json(undefined=Undefined.RAISE)
@dataclass(eq=True, kw_only=True, order=True, frozen=True)
class NewAxisBlkCfg(OptunaParameterable, DataClassJsonMixin):
    axis: int = -1

    @typing_extensions.override
    def optuna_params(
        self, trial: optuna.Trial, optuna_cfg: OptunaSearchCfg, prefix: str = ""
    ) -> "NewAxisBlkCfg":
        return self


class NewAxisBlk(nn.Module):
    def apply_array(self, _key: str, x: Array, _train: bool) -> Array:
        return jnp.expand_dims(x, axis=self.blk_cfg.axis)

    @nn.compact
    def __call__(self, inp: JaxArrayOrMap, train: bool) -> JaxArrayOrMap:
        return apply_arrayormap(inp, train, self.apply_array)


class NoopBlk(nn.Module):
    blk_cfg: EmptyBlkCfg

    @nn.compact
    def __call__(self, inp: JaxArrayOrMap, _train: bool) -> JaxArrayOrMap:
        return inp


@dataclass_json(undefined=Undefined.RAISE)
@dataclass(eq=True, kw_only=True, order=True, frozen=True)
class ReshapeBlkCfg(OptunaParameterable, DataClassJsonMixin):
    shape: tuple[int, ...] = (-1,)

    @typing_extensions.override
    def optuna_params(
        self, trial: optuna.Trial, optuna_cfg: OptunaSearchCfg, prefix: str = ""
    ) -> "ReshapeBlkCfg":
        return self


class ReshapeBlk(nn.Module):
    blk_cfg: ReshapeBlkCfg

    def apply_array(self, _key: str, x: Array, _train: bool) -> Array:
        return x.reshape(self.blk_cfg.shape)

    @nn.compact
    def __call__(self, inp: JaxArrayOrMap, train: bool) -> JaxArrayOrMap:
        return apply_arrayormap(inp, train, self.apply_array)


@dataclass_json(undefined=Undefined.RAISE)
@dataclass(eq=True, kw_only=True, order=True, frozen=True)
class MaskBlkCfg(OptunaParameterable, DataClassJsonMixin):
    mask: tuple[float, ...] = ()

    @typing_extensions.override
    def optuna_params(
        self, trial: optuna.Trial, optuna_cfg: OptunaSearchCfg, prefix: str = ""
    ) -> "MaskBlkCfg":
        return self


class MaskBlk(nn.Module):
    blk_cfg: MaskBlkCfg

    def apply_array(self, _key: str, x: Array, _train: bool) -> Array:
        return x * jnp.array(self.blk_cfg.mask)

    @nn.compact
    def __call__(self, inp: JaxArrayOrMap, train: bool) -> JaxArrayOrMap:
        return apply_arrayormap(inp, train, self.apply_array)
