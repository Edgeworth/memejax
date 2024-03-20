import copy
import dataclasses
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import optuna
import typing_extensions
from dataclasses_json import DataClassJsonMixin, Undefined, dataclass_json
from jinja2 import Template

from memejax.jax.hyperparam.trial import OptunaParameterable, OptunaSearchCfg

Variable = Any


@dataclass_json(undefined=Undefined.RAISE)
@dataclass(eq=True, kw_only=True, order=True)
class ModelCfg(OptunaParameterable, DataClassJsonMixin):
    variables: dict[str, Any] = field(default_factory=dict)
    dropout: float = 0.5
    input_dropout: float = 0.2
    layer_norm: bool = True

    def set_variable(self, name: str, value: Any) -> None:
        d = self.variables
        segments = name.split(".")
        for segment in segments[:-1]:
            d.setdefault(segment, {})
            d = d[segment]
        d[segments[-1]] = value

    def resolve(self, v: Variable) -> Any:
        if not isinstance(v, str):
            return v
        template = Template(v)
        return template.render(self.variables)

    def resolve_int(self, v: Variable) -> int:
        return int(self.resolve(v))

    @typing_extensions.override
    def optuna_params(
        self, trial: optuna.Trial, optuna_cfg: OptunaSearchCfg, prefix: str = ""
    ) -> "ModelCfg":
        cfg = copy.deepcopy(self)
        if optuna_cfg.model_params:
            cfg = dataclasses.replace(
                cfg,
                dropout=trial.suggest_float(prefix + "dropout", 0.0, 1.0, step=0.05),
                input_dropout=trial.suggest_float(prefix + "input_dropout", 0.0, 1.0, step=0.05),
                layer_norm=trial.suggest_categorical(prefix + "layer_norm", [True, False]),
            )
        return cfg


@dataclass_json(undefined=Undefined.RAISE)
@dataclass(eq=True, kw_only=True, order=True, frozen=True)
class EmptyBlkCfg(OptunaParameterable, DataClassJsonMixin): ...


class BlkKind(StrEnum):
    COMPOUND = "compound"
    CONCAT = "concat"
    CONV = "conv"
    ENCODER = "encoder"
    GAUSSIAN_NOISE = "gaussian_noise"
    LOSS_CROSS_ENTROPY_INTEGER = "loss_cross_entropy_integer"
    LOSS_MSE = "loss_mse"
    LOSS_KL_DIVERGENCE = "loss_kl_divergence"
    AXIS_OP = "axis_op"
    MASK = "mask"
    CONCAT_INPUTS = "concat_inputs"
    MEAN = "mean"
    MLP = "mlp"
    NOOP = "noop"
    POSEMB = "posemb"
    RESHAPE = "reshape"
    SAVED_CKPT = "saved_ckpt"
    SAVED_MODEL = "saved_model"
    SOFTMAX = "softmax"
    STACK_INPUTS = "stack_inputs"
    TRANSFORMER = "transformer"
