import copy
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import optuna
import typing_extensions
from dataclasses_json import DataClassJsonMixin, Undefined, dataclass_json

from memejax.jax.hyperparam.trial import OptunaParameterable, OptunaSearchCfg, suggest_enum


class XgbBoosterKind(StrEnum):
    GBTREE = "gbtree"
    GBLINEAR = "gblinear"
    DART = "dart"


class XgbTreeMethod(StrEnum):
    AUTO = "auto"
    EXACT = "exact"
    APPROX = "approx"
    HIST = "hist"


@dataclass_json(undefined=Undefined.RAISE)
@dataclass(eq=True, kw_only=True, order=True)
class XgbTrainCfg(OptunaParameterable, DataClassJsonMixin):
    booster: XgbBoosterKind = XgbBoosterKind.GBTREE
    lr: float = 0.3
    min_split_loss: float = 0.0
    max_depth: int = 6
    min_child_weight: float = 1.0
    max_delta_step: float = 0.0
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    l1_reg: float = 0.0
    l2_reg: float = 1.0
    tree_method: XgbTreeMethod = XgbTreeMethod.AUTO

    @typing_extensions.override
    def optuna_params(
        self, trial: optuna.Trial, optuna_cfg: OptunaSearchCfg, prefix: str = ""
    ) -> "XgbTrainCfg":
        cfg = copy.deepcopy(self)
        if optuna_cfg.optimizer_details:
            cfg.booster = suggest_enum(trial, prefix + "kind", XgbBoosterKind)
            cfg.tree_method = suggest_enum(trial, prefix + "tree_method", XgbTreeMethod)

        if optuna_cfg.learning_rates:
            cfg.lr = trial.suggest_float(prefix + "lr", 0.0, 1.0, step=0.01)
            cfg.subsample = trial.suggest_float(prefix + "subsample", 0.0, 1.0, step=0.01)
            cfg.colsample_bytree = trial.suggest_float(
                prefix + "colsample_bytree", 0.01, 1.0, step=0.01
            )

        if optuna_cfg.model_params:
            cfg.min_split_loss = trial.suggest_float(
                prefix + "min_split_loss", 0.0, 1000.0, log=True
            )
            cfg.min_child_weight = trial.suggest_float(
                prefix + "min_child_weight", 0.0, 1000.0, log=True
            )
            cfg.max_delta_step = trial.suggest_float(
                prefix + "max_delta_step", 0.0, 1000.0, log=True
            )

        if optuna_cfg.model_dims:
            cfg.max_depth = trial.suggest_int(prefix + "max_depth", 1, 15)

        if optuna_cfg.regularization_details:
            cfg.l1_reg = trial.suggest_float(prefix + "l1_reg", 0.0, 1000.0, log=True)
            cfg.l2_reg = trial.suggest_float(prefix + "l2_reg", 0.0, 1000.0, log=True)

        return cfg

    def make_params(self) -> dict[str, Any]:
        return {
            "booster": str(self.booster),
            "learning_rate": self.lr,
            "min_split_loss": self.min_split_loss,
            "max_depth": self.max_depth,
            "min_child_weight": self.min_child_weight,
            "max_delta_step": self.max_delta_step,
            "subsample": self.subsample,
            "reg_alpha": self.l1_reg,
            "reg_lambda": self.l2_reg,
            "tree_method": str(self.tree_method),
        }
