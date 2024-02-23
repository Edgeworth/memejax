import copy
import dataclasses
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path

import optax
import optuna
import typing_extensions
from dataclasses_json import DataClassJsonMixin, Exclude, Undefined, config, dataclass_json
from optax import ScalarOrSchedule

from memejax.hyperparam.trial import (
    OptunaParameterable,
    OptunaSearchCfg,
    suggest_enum,
    suggest_float_exp,
)


class OptimizerKind(StrEnum):
    ADABELIEF = "adabelief"
    ADAGRAD = "adagrad"
    ADAM = "adam"
    ADAMAX = "adamax"
    ADAMAXW = "adamaxw"
    ADAMW = "adamw"
    AMSGRAD = "amsgrad"
    FROMAGE = "fromage"
    LAMB = "lamb"
    LION = "lion"
    NADAM = "nadam"
    NADAMW = "nadamw"
    NOVOGRAD = "novograd"
    RADAM = "radam"
    RMSPROP = "rmsprop"
    SGD = "sgd"
    SM3 = "sm3"
    YOGI = "yogi"


class RegularizationKind(StrEnum):
    NONE = "none"
    L1 = "l1"
    L2 = "l2"
    ELASTIC = "elastic"


class ScheduleKind(StrEnum):
    CONSTANT = "constant"
    COSINE = "cosine"
    EXPONENTIAL = "exponential"
    POLYNOMIAL = "polynomial"


@dataclass_json(undefined=Undefined.RAISE)
@dataclass(eq=True, kw_only=True, order=True, frozen=True)
class OptimizerCfg(OptunaParameterable, DataClassJsonMixin):
    kind: OptimizerKind = OptimizerKind.ADABELIEF

    reg: RegularizationKind = RegularizationKind.NONE

    reg_lambda: float = 1e-4

    reg_alpha: float = 0.5
    """Only used if regularization is elastic"""

    adaptive_grad_clip: float | None = None
    """Adaptively clip gradient to this value if set"""

    schedule: ScheduleKind = ScheduleKind.CONSTANT

    decay_steps: int = 100000

    decay_end_lr: float = 1e-8

    decay_polynomial: float = 2.0
    """Only used if schedule is polynomial"""

    lr: float = 1e-5

    b1: float = 0.9

    b2: float = 0.999

    @typing_extensions.override
    def optuna_params(
        self, trial: optuna.Trial, optuna_cfg: OptunaSearchCfg, prefix: str = ""
    ) -> "OptimizerCfg":
        cfg = copy.deepcopy(self)

        if optuna_cfg.learning_rates:
            lr = suggest_float_exp(trial, prefix + "lr", 1e-9, 1e-1, base=10)
            decay_end_lr = suggest_float_exp(trial, prefix + "decay_end_lr", 1e-9, 1e-1, base=10)
            decay_polynomial = trial.suggest_float(prefix + "decay_polynomial", 1.0, 4.0, step=0.1)

            # Using a dynamic range for suggest_float seems to work poorly with the
            # dashboard.
            decay_end_lr = min(decay_end_lr, lr)

            cfg = dataclasses.replace(
                cfg,
                lr=lr,
                decay_end_lr=decay_end_lr,
                decay_polynomial=decay_polynomial,
                b1=trial.suggest_float(prefix + "b1", 0.0, 1.0, step=0.01),
                b2=trial.suggest_float(prefix + "b2", 0.0, 1.0, step=0.01),
            )

        if optuna_cfg.optimizer_details:
            cfg = dataclasses.replace(
                cfg,
                kind=suggest_enum(trial, prefix + "kind", OptimizerKind),
                schedule=suggest_enum(trial, prefix + "schedule", ScheduleKind),
            )

        if optuna_cfg.regularization_details:
            use_adaptive_grad_clip = trial.suggest_categorical(
                prefix + "use_adaptive_grad_clip", [False, True]
            )
            adaptive_grad_clip = None
            if use_adaptive_grad_clip:
                adaptive_grad_clip = trial.suggest_float(
                    prefix + "adaptive_grad_clip", 0.0, 1.0, step=0.01
                )
            cfg = dataclasses.replace(
                cfg,
                reg=suggest_enum(trial, prefix + "reg", RegularizationKind),
                reg_lambda=suggest_float_exp(trial, prefix + "reg_lambda", 1e-9, 1e-1, base=10),
                reg_alpha=trial.suggest_float(prefix + "reg_alpha", 0.0, 1.0, step=0.01),
                adaptive_grad_clip=adaptive_grad_clip,
            )

        return cfg

    def get_schedule(self) -> ScalarOrSchedule:
        match self.schedule:
            case ScheduleKind.CONSTANT:
                return self.lr
            case ScheduleKind.COSINE:
                warmup_steps = 0.05 * self.decay_steps
                return optax.warmup_cosine_decay_schedule(
                    # Also start with the final learning rate.
                    init_value=self.decay_end_lr,
                    peak_value=self.lr,
                    warmup_steps=warmup_steps,
                    decay_steps=self.decay_steps,
                    end_value=self.decay_end_lr,
                )
            case ScheduleKind.EXPONENTIAL:
                warmup_steps = 0.05 * self.decay_steps
                decay_rate = self.decay_end_lr / self.lr
                transition_steps = self.decay_steps - warmup_steps
                return optax.warmup_exponential_decay_schedule(
                    init_value=self.decay_end_lr,
                    peak_value=self.lr,
                    warmup_steps=warmup_steps,
                    transition_steps=transition_steps,
                    decay_rate=decay_rate,
                    transition_begin=0,
                    staircase=False,
                    end_value=self.decay_end_lr,
                )
            case ScheduleKind.POLYNOMIAL:
                return optax.polynomial_schedule(
                    init_value=self.lr,
                    end_value=self.decay_end_lr,
                    power=self.decay_polynomial,
                    transition_steps=self.decay_steps,
                )
        raise ValueError(f"Unknown schedule {self.schedule}")

    def get_optimizer(self) -> optax.GradientTransformation:
        schedule = self.get_schedule()
        match self.kind:
            case OptimizerKind.ADABELIEF:
                return optax.adabelief(schedule, b1=self.b1, b2=self.b2)
            case OptimizerKind.ADAGRAD:
                return optax.adagrad(schedule, initial_accumulator_value=self.b1)
            case OptimizerKind.ADAM:
                return optax.adam(schedule, b1=self.b1, b2=self.b2)
            case OptimizerKind.ADAMAX:
                return optax.adamax(schedule, b1=self.b1, b2=self.b2)
            case OptimizerKind.ADAMAXW:
                return optax.adamaxw(schedule, b1=self.b1, b2=self.b2)
            case OptimizerKind.ADAMW:
                return optax.adamw(schedule, b1=self.b1, b2=self.b2)
            case OptimizerKind.AMSGRAD:
                return optax.amsgrad(schedule, b1=self.b1, b2=self.b2)
            case OptimizerKind.FROMAGE:
                if not isinstance(schedule, float):
                    print(f"FROMAGE does not support schedules - using rate {self.lr}")
                return optax.fromage(self.lr)
            case OptimizerKind.LAMB:
                return optax.lamb(schedule, b1=self.b1, b2=self.b2)
            case OptimizerKind.LION:
                return optax.lion(schedule, b1=self.b1, b2=self.b2)
            case OptimizerKind.NADAM:
                return optax.nadam(schedule, b1=self.b1, b2=self.b2)
            case OptimizerKind.NADAMW:
                return optax.nadamw(schedule, b1=self.b1, b2=self.b2)
            case OptimizerKind.NOVOGRAD:
                return optax.novograd(schedule, b1=self.b1, b2=self.b2)
            case OptimizerKind.RADAM:
                return optax.radam(schedule, b1=self.b1, b2=self.b2)
            case OptimizerKind.RMSPROP:
                return optax.rmsprop(schedule, decay=self.b1)
            case OptimizerKind.SGD:
                return optax.sgd(schedule, momentum=self.b1)
            case OptimizerKind.SM3:
                if not isinstance(schedule, float):
                    print(f"SM3 does not support schedules - using rate {self.lr}")
                return optax.sm3(self.lr, momentum=self.b1)
            case OptimizerKind.YOGI:
                return optax.yogi(schedule, b1=self.b1, b2=self.b2)
        raise ValueError(f"Unknown optimizer {self.kind}")


@dataclass_json(undefined=Undefined.RAISE)
@dataclass(eq=True, kw_only=True, order=True)
class TrainCfg(DataClassJsonMixin):
    """Encapsulates all configuration for running training for a model."""

    name: str = field(init=False)
    """Automatically generated name of the config"""

    model_name: str
    """Name of the model"""

    output_path: Path = field(default_factory=Path, metadata=config(exclude=Exclude.ALWAYS))
    """Path to save model"""

    batch_size: int = 128
    """Batch size for training"""

    epoch_batches: int = -1
    """How many batches to run training on per epoch. -1 for all"""

    valid_batches: int = -1
    """How many batches to run validation. -1 for all"""

    report_secs: float | None = 0.5
    """After how many seconds to write out data to tensorboard"""

    report_batches: int | None = None
    """After how many batches to write out data to tensorboard"""

    print_secs: float | None = 5.0
    """After how many seconds to print summary data to console"""

    print_batches: int | None = None
    """After how many batches to print summary data to console"""

    print_grad_range_epochs: int | None = False
    """Print the range of gradients every given number of epochs"""

    check_epochs: int | None = 10
    """Check for various issues with training every given number of epochs"""

    report_epochs: int | None = 10
    """Report expensive info about the training process every given number of epochs"""

    ckpt_max_secs: float | None = 60.0 * 60.0
    """Maximum time between checkpoints"""

    opt_cfg: OptimizerCfg = field(default_factory=OptimizerCfg)

    def __post_init__(self) -> None:
        # Don't include anything except the model name and time, to allow
        # a model to be trained with multiple configs (using checkpointing).
        time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.name = f"{self.model_name}-{time_str}"

        if self.output_path == Path():
            self.output_path = Path("/tmp/memejax") / self.name  # noqa: S108
