from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, override

import flax.linen as nn
import optuna
from dataclasses_json import DataClassJsonMixin, Exclude, Undefined, config, dataclass_json

from memejax.jax.hyperparam.trial import OptunaParameterable, OptunaSearchCfg
from memejax.jax.pipeline.checkpoint import select_checkpoint
from memejax.jax.util import JaxArrayOrMap, resolve_path


@dataclass_json(undefined=Undefined.RAISE)
@dataclass(eq=True, kw_only=True, order=True, frozen=True)
class SavedCkptBlkCfg(OptunaParameterable, DataClassJsonMixin):
    path: Path
    output_key: str = "x"
    ckpt: Any = field(init=False, metadata=config(exclude=Exclude.ALWAYS))

    def __post_init__(self) -> None:
        path = resolve_path(self.path)
        assert path.is_dir(), f"File not found: {path}"
        object.__setattr__(self, "ckpt", select_checkpoint(path=path, best=True))

    @override
    def optuna_params(
        self, trial: optuna.Trial, optuna_cfg: OptunaSearchCfg, prefix: str = ""
    ) -> "SavedCkptBlkCfg":
        return self


class SavedCkptBlk(nn.Module):
    blk_cfg: SavedCkptBlkCfg

    @nn.compact
    def __call__(self, _inp: JaxArrayOrMap, _train: bool) -> JaxArrayOrMap:
        # TODO(-1): implement
        return _inp
