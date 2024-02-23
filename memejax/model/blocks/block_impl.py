from enum import StrEnum
from typing import TypeAlias

import flax.linen as nn
from dataclasses_json import DataClassJsonMixin

from memejax.model.blocks.cfg import BlkKind, EmptyBlkCfg
from memejax.model.blocks.compound import CompoundBlk, CompoundBlkCfg
from memejax.model.blocks.conv import ConvBlk, ConvBlkCfg
from memejax.model.blocks.encoder import EncoderBlk, EncoderBlkCfg
from memejax.model.blocks.loss import (
    LossCrossEntropyIntegerBlk,
    LossKLDivergenceBlk,
    LossLogPredictionCfg,
    LossMSEBlk,
    LossPredictionCfg,
)
from memejax.model.blocks.manipulation import (
    ConcatBlk,
    ConcatBlkCfg,
    NewAxisBlk,
    NewAxisBlkCfg,
    NoopBlk,
    ReshapeBlk,
    ReshapeBlkCfg,
    SoftmaxBlk,
)
from memejax.model.blocks.mlp import MlpBlk, MlpBlkCfg
from memejax.model.blocks.noise import GaussianNoiseBlk, GaussianNoiseBlkCfg
from memejax.model.blocks.posemb import PosEmbBlk, PosEmbBlkCfg
from memejax.model.blocks.transformer import TransformerBlk, TransformerBlkCfg

BlkCfgType: TypeAlias = (
    CompoundBlkCfg
    | ConcatBlkCfg
    | ConvBlkCfg
    | EmptyBlkCfg
    | EncoderBlkCfg
    | GaussianNoiseBlkCfg
    | LossPredictionCfg
    | LossLogPredictionCfg
    | MlpBlkCfg
    | NewAxisBlkCfg
    | PosEmbBlkCfg
    | ReshapeBlkCfg
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
    BlkKind.NOOP: NoopBlk,
    BlkKind.NEWAXIS: NewAxisBlk,
    BlkKind.POSEMB: PosEmbBlk,
    BlkKind.RESHAPE: ReshapeBlk,
    BlkKind.SOFTMAX: SoftmaxBlk,
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
    BlkKind.NOOP: EmptyBlkCfg,
    BlkKind.NEWAXIS: NewAxisBlkCfg,
    BlkKind.POSEMB: PosEmbBlkCfg,
    BlkKind.RESHAPE: ReshapeBlkCfg,
    BlkKind.SOFTMAX: EmptyBlkCfg,
    BlkKind.TRANSFORMER: TransformerBlkCfg,
}
