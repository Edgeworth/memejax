import copy
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import optuna
import typing_extensions
from dataclasses_json import DataClassJsonMixin, Undefined, dataclass_json

from memejax.jax.hyperparam.trial import (
    OptunaParameterable,
    OptunaSearchCfg,
    suggest_enum,
    suggest_float_exp,
)


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
    kind: XgbBoosterKind = XgbBoosterKind.GBTREE
    tree_method: XgbTreeMethod = XgbTreeMethod.AUTO
    lr: float = 0.3

    # Minimum loss reduction required to make a further partition. Higher makes the model more
    # conservative (less overfitting).
    min_split_loss: float = 0.0

    # Maximum depth of tree. Higher makes the model more complex and likely to overfit.
    max_depth: int = 6

    # Minimum sum of hessian required in a child. Higher makes the model more conservative (less
    # overfitting).
    min_child_weight: float = 1.0

    # Higher makes the model more conservative (less overfitting). 0.0 means no constraint.
    max_delta_step: float = 0.0

    # Random sampling proportion of training data. Lower means less overfitting.
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    colsample_bylevel: float = 1.0
    colsample_bynode: float = 1.0
    l1_reg: float = 0.0
    l2_reg: float = 1.0

    @typing_extensions.override
    def optuna_params(
        self, trial: optuna.Trial, optuna_cfg: OptunaSearchCfg, prefix: str = ""
    ) -> "XgbTrainCfg":
        cfg = copy.deepcopy(self)
        if optuna_cfg.optimizer_details:
            cfg.kind = suggest_enum(trial, prefix + "kind", XgbBoosterKind)
            cfg.tree_method = suggest_enum(trial, prefix + "tree_method", XgbTreeMethod)

        if optuna_cfg.learning_rates:
            # learning rate: [0.01, 0.1, 1]
            cfg.lr = trial.suggest_float(prefix + "lr", 1e-3, 1.0, log=True)
            # subsample: [0.1, 0.5, 0.9]
            cfg.subsample = trial.suggest_float(prefix + "subsample", 0.1, 1.0, step=0.1)
            # colsample_bytree: [0.1, 0.5, 0.9]
            cfg.colsample_bytree = trial.suggest_float(
                prefix + "colsample_bytree", 0.1, 1.0, step=0.1
            )
            cfg.colsample_bylevel = trial.suggest_float(
                prefix + "colsample_bylevel", 0.1, 1.0, step=0.1
            )
            cfg.colsample_bynode = trial.suggest_float(
                prefix + "colsample_bynode", 0.1, 1.0, step=0.1
            )

        if optuna_cfg.model_params:
            # gamma: [0, 1, 10]
            cfg.min_split_loss = trial.suggest_float(
                prefix + "min_split_loss", 0.0, 10.0, step=0.01
            )
            # min_child_weight: [1.0, 5.0, 10.0]
            cfg.min_child_weight = trial.suggest_float(
                prefix + "min_child_weight", 0.0, 1000.0, step=0.01
            )
            cfg.max_delta_step = trial.suggest_float(
                prefix + "max_delta_step", 0.0, 10.0, step=0.01
            )

        if optuna_cfg.model_dims:
            # max depth
            cfg.max_depth = trial.suggest_int(prefix + "max_depth", 1, 30)

        if optuna_cfg.regularization_details:
            # alpha
            cfg.l1_reg = suggest_float_exp(trial, prefix + "l1_reg", 1e-9, 1000.0, base=10)
            # lambda
            cfg.l2_reg = suggest_float_exp(trial, prefix + "l2_reg", 1e-9, 1000.0, base=10)

        return cfg

    def make_params(self) -> dict[str, Any]:
        return {
            "booster": str(self.kind),
            "learning_rate": self.lr,
            "min_split_loss": self.min_split_loss,
            "max_depth": self.max_depth,
            "min_child_weight": self.min_child_weight,
            "max_delta_step": self.max_delta_step,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "colsample_bylevel": self.colsample_bylevel,
            "colsample_bynode": self.colsample_bynode,
            "reg_alpha": self.l1_reg,
            "reg_lambda": self.l2_reg,
            "tree_method": str(self.tree_method),
        }
