from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, TypeVar

import cloup
import optuna
from dataclasses_json import DataClassJsonMixin, Undefined, dataclass_json

from memejax.common import ceil_log, floor_log


@dataclass_json(undefined=Undefined.RAISE)
@dataclass(eq=True, kw_only=True, order=True, frozen=True)
class OptunaSearchCfg(DataClassJsonMixin):
    model_dims: bool
    model_params: bool
    learning_rates: bool
    optimizer_details: bool
    regularization_details: bool
    sample_size: int


optuna_search_cfg_options = cloup.option_group(
    "Optuna search config options",
    cloup.option(
        "--optuna-model-dims/--no-optuna-model-dims",
        is_flag=True,
        default=False,
        help="Search for architecture dimensions - very slow.",
    ),
    cloup.option(
        "--optuna-model-params/--no-optuna-model-params",
        is_flag=True,
        default=False,
        help="Search for model params.",
    ),
    cloup.option(
        "--optuna-learning-rates/--no-optuna-learning-rates",
        is_flag=True,
        default=False,
        help="Search for learning rates.",
    ),
    cloup.option(
        "--optuna-optimizer-details/--no-optuna-optimizer-details",
        is_flag=True,
        default=False,
        help="Search for optimizer details.",
    ),
    cloup.option(
        "--optuna-regularization-details/--no-optuna-regularization-details",
        is_flag=True,
        default=False,
        help="Search for regularization details.",
    ),
    cloup.option(
        "--optuna-all", is_flag=True, default=False, help="Search for regularization details."
    ),
    cloup.option(
        "--optuna-sample-size",
        type=int,
        default=1,
        help="Number of parallel trainers to run and average for each trial.",
    ),
)


def build_optuna_search_cfg_from_args(
    optuna_model_dims: bool,
    optuna_model_params: bool,
    optuna_learning_rates: bool,
    optuna_optimizer_details: bool,
    optuna_regularization_details: bool,
    optuna_all: bool,
    optuna_sample_size: int,
    **_kwargs: Any,
) -> OptunaSearchCfg:
    if optuna_all:
        assert not any(
            [
                optuna_model_dims,
                optuna_model_params,
                optuna_learning_rates,
                optuna_optimizer_details,
                optuna_regularization_details,
            ]
        )
        optuna_model_dims = True
        optuna_model_params = True
        optuna_learning_rates = True
        optuna_optimizer_details = True
        optuna_regularization_details = True

    return OptunaSearchCfg(
        model_dims=optuna_model_dims,
        model_params=optuna_model_params,
        learning_rates=optuna_learning_rates,
        optimizer_details=optuna_optimizer_details,
        regularization_details=optuna_regularization_details,
        sample_size=optuna_sample_size,
    )


class OptunaParameterable(Protocol):
    def optuna_params(
        self, trial: optuna.Trial, optuna_cfg: OptunaSearchCfg, prefix: str = ""
    ) -> Any:
        ...


_T = TypeVar("_T", bound=StrEnum)


def suggest_enum(trial: optuna.Trial, name: str, enum_class: type[_T]) -> _T:
    selected_option = trial.suggest_categorical(name, [str(i) for i in enum_class])
    return enum_class(selected_option)


def suggest_int_exp(
    trial: optuna.Trial, name: str, min_value: int, max_value: int, base: int
) -> int:
    name = f"{name}_log{base}"
    min_value = floor_log(min_value, base)
    max_value = ceil_log(max_value, base)
    exp = trial.suggest_int(name, min_value, max_value)
    assert exp >= 0
    return int(base**exp)


def suggest_float_exp(
    trial: optuna.Trial, name: str, min_value: float, max_value: float, base: int
) -> float:
    name = f"{name}_log{base}"
    min_value = floor_log(min_value, base)
    max_value = ceil_log(max_value, base)
    exp = trial.suggest_int(name, min_value, max_value)
    return float(float(base) ** float(exp))
