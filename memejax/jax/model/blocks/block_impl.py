from enum import StrEnum
from typing import TypeAlias

import flax.linen as nn
from dataclasses_json import DataClassJsonMixin

from memejax.jax.model.blocks.cfg import BlkKind, EmptyBlkCfg
from memejax.jax.model.blocks.compound import CompoundBlk, CompoundBlkCfg
from memejax.jax.model.blocks.conv import ConvBlk, ConvBlkCfg
from memejax.jax.model.blocks.encoder import EncoderBlk, EncoderBlkCfg
from memejax.jax.model.blocks.loss import (
    LossCrossEntropyIntegerBlk,
    LossKLDivergenceBlk,
    LossLogPredictionCfg,
    LossMSEBlk,
    LossPredictionCfg,
)
from memejax.jax.model.blocks.manipulation import (
    AxisOpBlk,
    AxisOpBlkCfg,
    BinOpBlk,
    BinOpBlkCfg,
    ConcatBlk,
    ConcatBlkCfg,
    ConcatInputsBlk,
    MapInputsBlkCfg,
    MaskBlk,
    MaskBlkCfg,
    NoopBlk,
    ReshapeBlk,
    ReshapeBlkCfg,
    SoftmaxBlk,
    StackInputsBlk,
)
from memejax.jax.model.blocks.mlp import MlpBlk, MlpBlkCfg
from memejax.jax.model.blocks.noise import GaussianNoiseBlk, GaussianNoiseBlkCfg
from memejax.jax.model.blocks.posemb import PosEmbBlk, PosEmbBlkCfg
from memejax.jax.model.blocks.saved_ckpt import SavedCkptBlk, SavedCkptBlkCfg
from memejax.jax.model.blocks.saved_model import SavedModelBlk, SavedModelBlkCfg
from memejax.jax.model.blocks.transformer import TransformerBlk, TransformerBlkCfg

BlkCfgType: TypeAlias = (
    CompoundBlkCfg
    | ConcatBlkCfg
    | ConvBlkCfg
    | EmptyBlkCfg
    | EncoderBlkCfg
    | GaussianNoiseBlkCfg
    | LossPredictionCfg
    | LossLogPredictionCfg
    | MaskBlkCfg
    | MapInputsBlkCfg
    | MlpBlkCfg
    | AxisOpBlkCfg
    | BinOpBlkCfg
    | PosEmbBlkCfg
    | ReshapeBlkCfg
    | SavedCkptBlkCfg
    | SavedModelBlkCfg
    | TransformerBlkCfg
)


def blk_kind_cfg_class(self: BlkKind) -> type[BlkCfgType]:
    return _BLKKIND_TO_BLK_CFG_CLS[self]


def blk_kind_model_class(self: BlkKind) -> type[nn.Module]:
    return _BLKKIND_TO_BLK_CLS[self]


def _add_strenum_variant(cls: type[StrEnum], name: str, val_name: str) -> None:
    val = str.__new__(cls, val_name)
    val._name_ = name
    val._value_ = val_name
    cls._member_names_.append(name)
    cls._member_map_[name] = val
    cls._value2member_map_[val] = cls._member_map_[name]


def append_blk_kind(name: str, blk_class: type, blk_cfg_class: type[DataClassJsonMixin]) -> BlkKind:
    _add_strenum_variant(BlkKind, name.upper(), name)
    kind = BlkKind(name)
    _BLKKIND_TO_BLK_CLS[kind] = blk_class
    _BLKKIND_TO_BLK_CFG_CLS[kind] = blk_cfg_class  # type: ignore[assignment]
    return kind


_BLKKIND_TO_BLK_CLS = {
    BlkKind.COMPOUND: CompoundBlk,
    BlkKind.CONCAT: ConcatBlk,
    BlkKind.CONV: ConvBlk,
    BlkKind.ENCODER: EncoderBlk,
    BlkKind.GAUSSIAN_NOISE: GaussianNoiseBlk,
    BlkKind.LOSS_CROSS_ENTROPY_INTEGER: LossCrossEntropyIntegerBlk,
    BlkKind.LOSS_MSE: LossMSEBlk,
    BlkKind.LOSS_KL_DIVERGENCE: LossKLDivergenceBlk,
    BlkKind.MLP: MlpBlk,
    BlkKind.MASK: MaskBlk,
    BlkKind.CONCAT_INPUTS: ConcatInputsBlk,
    BlkKind.NOOP: NoopBlk,
    BlkKind.AXIS_OP: AxisOpBlk,
    BlkKind.BIN_OP: BinOpBlk,
    BlkKind.POSEMB: PosEmbBlk,
    BlkKind.RESHAPE: ReshapeBlk,
    BlkKind.SAVED_CKPT: SavedCkptBlk,
    BlkKind.SAVED_MODEL: SavedModelBlk,
    BlkKind.SOFTMAX: SoftmaxBlk,
    BlkKind.STACK_INPUTS: StackInputsBlk,
    BlkKind.TRANSFORMER: TransformerBlk,
}

_BLKKIND_TO_BLK_CFG_CLS: dict[BlkKind, type[BlkCfgType]] = {
    BlkKind.COMPOUND: CompoundBlkCfg,
    BlkKind.CONCAT: ConcatBlkCfg,
    BlkKind.CONV: ConvBlkCfg,
    BlkKind.ENCODER: EncoderBlkCfg,
    BlkKind.GAUSSIAN_NOISE: GaussianNoiseBlkCfg,
    BlkKind.LOSS_CROSS_ENTROPY_INTEGER: LossPredictionCfg,
    BlkKind.LOSS_MSE: LossPredictionCfg,
    BlkKind.LOSS_KL_DIVERGENCE: LossLogPredictionCfg,
    BlkKind.MLP: MlpBlkCfg,
    BlkKind.MASK: MaskBlkCfg,
    BlkKind.CONCAT_INPUTS: MapInputsBlkCfg,
    BlkKind.NOOP: EmptyBlkCfg,
    BlkKind.AXIS_OP: AxisOpBlkCfg,
    BlkKind.BIN_OP: BinOpBlkCfg,
    BlkKind.POSEMB: PosEmbBlkCfg,
    BlkKind.RESHAPE: ReshapeBlkCfg,
    BlkKind.SAVED_CKPT: SavedCkptBlkCfg,
    BlkKind.SAVED_MODEL: SavedModelBlkCfg,
    BlkKind.SOFTMAX: EmptyBlkCfg,
    BlkKind.STACK_INPUTS: MapInputsBlkCfg,
    BlkKind.TRANSFORMER: TransformerBlkCfg,
}
