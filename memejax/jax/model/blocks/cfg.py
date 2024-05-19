import copy
import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, override

import flax.linen as nn
import optuna
from dataclasses_json import DataClassJsonMixin, Undefined, dataclass_json
from jax import Array
from jax.typing import ArrayLike
from jinja2 import Template

from memejax.jax.hyperparam.trial import OptunaParameterable, OptunaSearchCfg

Variable = Any


class ActivationKind(StrEnum):
    CELU = "celu"
    ELU = "elu"
    GELU = "gelu"
    LEAKY_RELU = "leaky_relu"
    LOG_SIGMOID = "log_sigmoid"
    RELU = "relu"
    SIGMOID = "sigmoid"
    SILU = "silu"
    TANH = "tanh"

    def get_func(self) -> Callable[[ArrayLike], Array]:
        if self == ActivationKind.CELU:
            return nn.celu
        if self == ActivationKind.ELU:
            return nn.elu
        if self == ActivationKind.GELU:
            return nn.gelu
        if self == ActivationKind.LEAKY_RELU:
            return nn.leaky_relu
        if self == ActivationKind.LOG_SIGMOID:
            return nn.log_sigmoid
        if self == ActivationKind.RELU:
            return nn.relu
        if self == ActivationKind.SIGMOID:
            return nn.sigmoid
        if self == ActivationKind.SILU:
            return nn.silu
        if self == ActivationKind.TANH:
            return nn.tanh
        raise ValueError(f"Unknown activation kind: {self}")


@dataclass_json(undefined=Undefined.RAISE)
@dataclass(eq=True, kw_only=True, order=True)
class ModelCfg(OptunaParameterable, DataClassJsonMixin):
    variables: dict[str, Any] = field(default_factory=dict)
    dropout: float = 0.5
    input_dropout: float = 0.2
    layer_norm: bool = True
    activation: ActivationKind = ActivationKind.LEAKY_RELU

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

    @override
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
                activation=trial.suggest_categorical(prefix + "activation", list(ActivationKind)),
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
    BIN_OP = "bin_op"
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
