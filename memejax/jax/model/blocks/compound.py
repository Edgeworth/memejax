import copy
from dataclasses import dataclass, field
from functools import partial
from graphlib import TopologicalSorter
from typing import Any, override

import flax.linen as nn
import jax
import optuna
from dataclasses_json import DataClassJsonMixin, Undefined, config, dataclass_json
from flax.typing import VariableDict

from memejax.common import dataclass_has_field
from memejax.jax.hyperparam.trial import OptunaParameterable, OptunaSearchCfg
from memejax.jax.model.blocks.cfg import BlkKind, ModelCfg
from memejax.jax.pipeline.metrics import MetricCollection, MetricCollectionMap
from memejax.jax.util import JaxArrayMap


@dataclass_json(undefined=Undefined.RAISE)
@dataclass(eq=True, kw_only=True, order=True, frozen=True)
class CompoundInput(DataClassJsonMixin):
    src: str
    # TODO(0): Add | str when https://github.com/lidatong/dataclasses-json/issues/502 is fixed.
    cols: tuple[tuple[str, str], ...] = ()


@dataclass_json(undefined=Undefined.RAISE)
@dataclass(eq=True, kw_only=True, order=True, frozen=True)
class CompoundItem(DataClassJsonMixin):
    name: str
    blk_kind: BlkKind
    # Use Any here to work around circular imports.
    blk_cfg: Any = field(metadata=config(decoder=dict))
    inputs: tuple[CompoundInput, ...] = ()
    is_output: bool = False

    def __post_init__(self) -> None:
        from memejax.jax.model.blocks.block_impl import blk_kind_cfg_class  # noqa: PLC0415

        if isinstance(self.blk_cfg, dict):
            # Work around frozen, but we need hash.
            object.__setattr__(
                self, "blk_cfg", blk_kind_cfg_class(self.blk_kind).from_dict(self.blk_cfg)
            )


@dataclass_json(undefined=Undefined.RAISE)
@dataclass(eq=True, kw_only=True, order=True, frozen=True)
class CompoundBlkCfg(OptunaParameterable, DataClassJsonMixin):
    blks: tuple[CompoundItem, ...] = ()
    losses: tuple[CompoundItem, ...] = ()

    def get_blk_by_name(self, name: str) -> CompoundItem | None:
        for blk in self.blks:
            if blk.name == name:
                return blk
        return None

    @override
    def optuna_params(
        self, trial: optuna.Trial, optuna_cfg: OptunaSearchCfg, prefix: str = ""
    ) -> "CompoundBlkCfg":
        cfg = copy.deepcopy(self)
        for loss in cfg.blks:
            # Work around frozen, but we need hash.
            object.__setattr__(
                loss,
                "blk_cfg",
                loss.blk_cfg.optuna_params(trial, optuna_cfg, f"{prefix}{loss.name}/"),
            )
        for loss in cfg.losses:
            # Work around frozen, but we need hash.
            object.__setattr__(
                loss,
                "blk_cfg",
                loss.blk_cfg.optuna_params(trial, optuna_cfg, f"{prefix}{loss.name}/"),
            )
        return cfg


CompoundBlkData = dict[str, JaxArrayMap]


class CompoundBlk(nn.Module):
    cfg: ModelCfg
    blk_cfg: CompoundBlkCfg

    @staticmethod
    def _collect_inputs(results: CompoundBlkData, blk: CompoundItem) -> JaxArrayMap:
        inputs: JaxArrayMap = {}
        for inp in blk.inputs:
            src = results[inp.src]
            cols = src.keys() if len(inp.cols) == 0 else inp.cols
            for col in cols:
                if isinstance(col, tuple):
                    assert col[1] not in inputs, f"Duplicate column {col[1]}"
                    inputs[col[1]] = src[col[0]]
                else:
                    assert col not in inputs, f"Duplicate column {col}"
                    assert isinstance(col, str)
                    inputs[col] = src[col]
        return inputs

    @staticmethod
    @partial(jax.jit, static_argnames=("blk_cfg"))
    def metrics(
        blk_cfg: CompoundBlkCfg,
        variables: VariableDict,
        model_output: CompoundBlkData,
        aux: JaxArrayMap,
    ) -> MetricCollectionMap:
        from memejax.jax.model.blocks.block_impl import blk_kind_model_class  # noqa: PLC0415

        metrics = {}
        results = {**model_output, "aux": aux}
        assert len(results) == len(model_output) + 1, "Duplicate column for aux"
        for blk in blk_cfg.losses:
            inputs = CompoundBlk._collect_inputs(results, blk)
            module = blk_kind_model_class(blk.blk_kind)(blk_cfg=blk.blk_cfg)
            assert blk.name not in metrics, f"Duplicate metric collection {blk.name}"
            m = module.apply(variables, inputs)
            assert isinstance(m, MetricCollection), f"{blk.name} is not a metric collection"
            metrics[blk.name] = m
        return metrics

    @nn.compact
    def __call__(self, x: JaxArrayMap, train: bool) -> CompoundBlkData:
        from memejax.jax.model.blocks.block_impl import blk_kind_model_class  # noqa: PLC0415

        ts: TopologicalSorter = TopologicalSorter()
        blks = {blk.name: blk for blk in self.blk_cfg.blks}
        for blk in blks.values():
            ts.add(blk.name, *[i.src for i in blk.inputs])

        results: CompoundBlkData = {"input": x}
        outputs: CompoundBlkData = {}
        for name in ts.static_order():
            # no need to run anything for the input.
            if name == "input":
                continue
            blk = blks[name]
            inputs = CompoundBlk._collect_inputs(results, blk)
            blk_cls = blk_kind_model_class(blk.blk_kind)
            blk_cls_input = {}
            if dataclass_has_field(blk_cls, "cfg"):
                blk_cls_input["cfg"] = self.cfg
            if dataclass_has_field(blk_cls, "blk_cfg"):
                blk_cls_input["blk_cfg"] = blk.blk_cfg
            module = blk_cls(**blk_cls_input)
            results[name] = module(inputs, train)
            if blk.is_output:
                assert name not in outputs, f"Duplicate output {name}"
                outputs[name] = results[name]
        assert len(outputs) > 0, "No output found"
        return outputs
