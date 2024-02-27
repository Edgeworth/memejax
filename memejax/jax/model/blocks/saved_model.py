from dataclasses import dataclass, field
from pathlib import Path

import flax.linen as nn
import optuna
import typing_extensions
from dataclasses_json import DataClassJsonMixin, Exclude, Undefined, config, dataclass_json
from jax.experimental import jax2tf
from jax.experimental.jax2tf.call_tf import call_tf_p
from jax.interpreters import batching

from memejax.jax.hyperparam.trial import OptunaParameterable, OptunaSearchCfg
from memejax.jax.pipeline.inference import JaxSavedModelInference
from memejax.jax.util import JaxArrayOrMap, resolve_path


@dataclass_json(undefined=Undefined.RAISE)
@dataclass(eq=True, kw_only=True, order=True, frozen=True)
class SavedModelBlkCfg(OptunaParameterable, DataClassJsonMixin):
    path: Path
    sm: JaxSavedModelInference = field(init=False, metadata=config(exclude=Exclude.ALWAYS))

    def __post_init__(self) -> None:
        path = resolve_path(self.path)
        assert path.is_dir(), f"File not found: {path}"
        object.__setattr__(self, "sm", JaxSavedModelInference(path=path))

    @typing_extensions.override
    def optuna_params(
        self, trial: optuna.Trial, optuna_cfg: OptunaSearchCfg, prefix: str = ""
    ) -> "SavedModelBlkCfg":
        return self


class SavedModelBlk(nn.Module):
    blk_cfg: SavedModelBlkCfg

    @nn.compact
    def __call__(self, inp: JaxArrayOrMap, _train: bool) -> JaxArrayOrMap:
        assert len(inp) == 1
        inp_col = next(iter(inp.keys()))
        output = {inp_col: jax2tf.call_tf(self.blk_cfg.sm.tf_func())(inp)}
        print(output)
        return output


# Passthrough batcher for tf saved models which assumes it can take as many inputs
# as required on the first axis (shape polymorphic).
def _tf_passthrough_batcher(batched_args, batched_dims, **kwargs) -> tuple:
    ret = call_tf_p.bind(batched_args[0], **kwargs)
    # This assumes that the results are batched on the first dimension.
    return (ret, (0,) * len(ret))


batching.primitive_batchers[call_tf_p] = _tf_passthrough_batcher
