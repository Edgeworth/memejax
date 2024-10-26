from dataclasses import dataclass
from enum import StrEnum
from typing import override

import flax.linen as nn
import jax.numpy as jnp
import optuna
from dataclasses_json import DataClassJsonMixin, Undefined, dataclass_json
from jax import Array

from memejax.jax.hyperparam.trial import OptunaParameterable, OptunaSearchCfg
from memejax.jax.model.blocks.cfg import EmptyBlkCfg
from memejax.jax.util import JaxArrayMap, JaxArrayOrMap, apply_arrayormap


@dataclass_json(undefined=Undefined.RAISE)
@dataclass(eq=True, kw_only=True, order=True, frozen=True)
class MapInputsBlkCfg(OptunaParameterable, DataClassJsonMixin):
    output_key: str = "x"

    @override
    def optuna_params(
        self, trial: optuna.Trial, optuna_cfg: OptunaSearchCfg, prefix: str = ""
    ) -> "MapInputsBlkCfg":
        return self


class ConcatInputsBlk(nn.Module):
    blk_cfg: MapInputsBlkCfg

    @nn.compact
    def __call__(self, inp: JaxArrayMap, _train: bool) -> JaxArrayMap:
        # Merge all inputs into a single output.
        # Sort values by key to ensure consistent order.
        values = [inp[k] for k in sorted(inp.keys())]
        return {self.blk_cfg.output_key: jnp.concatenate(values, axis=0)}


class StackInputsBlk(nn.Module):
    blk_cfg: MapInputsBlkCfg

    @nn.compact
    def __call__(self, inp: JaxArrayMap, _train: bool) -> JaxArrayMap:
        # Merge all inputs into a single output.
        # Sort values by key to ensure consistent order.
        values = [inp[k] for k in sorted(inp.keys())]
        return {self.blk_cfg.output_key: jnp.stack(values)}


class SoftmaxBlk(nn.Module):
    def apply_array(self, _key: str, x: Array, _train: bool) -> Array:
        x = nn.softmax(x)
        return x

    @nn.compact
    def __call__[T: JaxArrayOrMap](self, inp: T, train: bool) -> T:
        return apply_arrayormap(inp, train, self.apply_array)


@dataclass_json(undefined=Undefined.RAISE)
@dataclass(eq=True, kw_only=True, order=True, frozen=True)
class ConcatBlkCfg(OptunaParameterable, DataClassJsonMixin):
    axis: int = -1
    value: float = 0.0
    times: int = 1

    @override
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
    def __call__[T: JaxArrayOrMap](self, inp: T, train: bool) -> T:
        return apply_arrayormap(inp, train, self.apply_array)


class AxisOp(StrEnum):
    # Differentiable:
    MEAN = "mean"
    SUM = "sum"
    NEW_AXIS = "new_axis"

    # Non-differentiable:
    MAX = "max"
    MIN = "min"


@dataclass_json(undefined=Undefined.RAISE)
@dataclass(eq=True, kw_only=True, order=True, frozen=True)
class AxisOpBlkCfg(OptunaParameterable, DataClassJsonMixin):
    op: AxisOp
    axis: int = -1

    @override
    def optuna_params(
        self, trial: optuna.Trial, optuna_cfg: OptunaSearchCfg, prefix: str = ""
    ) -> "AxisOpBlkCfg":
        return self


class AxisOpBlk(nn.Module):
    blk_cfg: AxisOpBlkCfg

    def apply_array(self, _key: str, x: Array, _train: bool) -> Array:
        match self.blk_cfg.op:
            case AxisOp.NEW_AXIS:
                return jnp.expand_dims(x, axis=self.blk_cfg.axis)
            case AxisOp.MEAN:
                return jnp.mean(x, axis=self.blk_cfg.axis)
            case AxisOp.SUM:
                return jnp.sum(x, axis=self.blk_cfg.axis)
            case AxisOp.MAX:
                return jnp.max(x, axis=self.blk_cfg.axis)
            case AxisOp.MIN:
                return jnp.min(x, axis=self.blk_cfg.axis)

    @nn.compact
    def __call__[T: JaxArrayOrMap](self, inp: T, train: bool) -> T:
        return apply_arrayormap(inp, train, self.apply_array)


class BinOp(StrEnum):
    MUL = "mul"
    DIV = "div"
    ADD = "add"
    SUB = "sub"
    EXP = "exp"
    LOG = "log"


@dataclass_json(undefined=Undefined.RAISE)
@dataclass(eq=True, kw_only=True, order=True, frozen=True)
class BinOpBlkCfg(OptunaParameterable, DataClassJsonMixin):
    op: BinOp
    value: float = 1.0

    @override
    def optuna_params(
        self, trial: optuna.Trial, optuna_cfg: OptunaSearchCfg, prefix: str = ""
    ) -> "BinOpBlkCfg":
        return self


class BinOpBlk(nn.Module):
    blk_cfg: BinOpBlkCfg

    def apply_array(self, _key: str, x: Array, _train: bool) -> Array:
        match self.blk_cfg.op:
            case BinOp.MUL:
                return x * self.blk_cfg.value
            case BinOp.DIV:
                return x / self.blk_cfg.value
            case BinOp.ADD:
                return x + self.blk_cfg.value
            case BinOp.SUB:
                return x - self.blk_cfg.value
            case BinOp.EXP:
                return x**self.blk_cfg.value
            case BinOp.LOG:
                return jnp.log(x) / jnp.log(self.blk_cfg.value)

    @nn.compact
    def __call__[T: JaxArrayOrMap](self, inp: T, train: bool) -> T:
        return apply_arrayormap(inp, train, self.apply_array)


class NoopBlk(nn.Module):
    blk_cfg: EmptyBlkCfg

    @nn.compact
    def __call__[T: JaxArrayOrMap](self, inp: T, _train: bool) -> T:
        return inp


@dataclass_json(undefined=Undefined.RAISE)
@dataclass(eq=True, kw_only=True, order=True, frozen=True)
class ReshapeBlkCfg(OptunaParameterable, DataClassJsonMixin):
    shape: tuple[int, ...] = (-1,)

    @override
    def optuna_params(
        self, trial: optuna.Trial, optuna_cfg: OptunaSearchCfg, prefix: str = ""
    ) -> "ReshapeBlkCfg":
        return self


class ReshapeBlk(nn.Module):
    blk_cfg: ReshapeBlkCfg

    def apply_array(self, _key: str, x: Array, _train: bool) -> Array:
        return x.reshape(self.blk_cfg.shape)

    @nn.compact
    def __call__[T: JaxArrayOrMap](self, inp: T, train: bool) -> T:
        return apply_arrayormap(inp, train, self.apply_array)


@dataclass_json(undefined=Undefined.RAISE)
@dataclass(eq=True, kw_only=True, order=True, frozen=True)
class MaskBlkCfg(OptunaParameterable, DataClassJsonMixin):
    mask: tuple[float, ...] = ()

    @override
    def optuna_params(
        self, trial: optuna.Trial, optuna_cfg: OptunaSearchCfg, prefix: str = ""
    ) -> "MaskBlkCfg":
        return self


class MaskBlk(nn.Module):
    blk_cfg: MaskBlkCfg

    def apply_array(self, _key: str, x: Array, _train: bool) -> Array:
        return x * jnp.array(self.blk_cfg.mask)

    @nn.compact
    def __call__[T: JaxArrayOrMap](self, inp: T, train: bool) -> T:
        return apply_arrayormap(inp, train, self.apply_array)
