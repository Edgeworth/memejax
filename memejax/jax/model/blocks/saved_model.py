import functools
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import flax.linen as nn
import jax
import numpy as np
import optuna
import tensorflow as tf
import typing_extensions
from dataclasses_json import DataClassJsonMixin, Exclude, Undefined, config, dataclass_json
from jax.experimental import jax2tf
from jax.experimental.jax2tf.call_tf import TfVal, call_tf_p
from jax.interpreters import batching

from memejax.jax.hyperparam.trial import OptunaParameterable, OptunaSearchCfg
from memejax.jax.pipeline.inference import JaxSavedModelInference
from memejax.jax.util import JaxArrayOrMap, resolve_path


@dataclass_json(undefined=Undefined.RAISE)
@dataclass(eq=True, kw_only=True, order=True, frozen=True)
class SavedModelBlkCfg(OptunaParameterable, DataClassJsonMixin):
    path: Path
    output_key: str = "x"
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
        fn = self.blk_cfg.sm.tf_func()
        batching.primitive_batchers[call_tf_p] = functools.partial(_tf_passthrough_batcher, fn, inp)
        output = {self.blk_cfg.output_key: jax2tf.call_tf(fn)(inp)}
        return output


@dataclass(eq=True, kw_only=True, order=True, frozen=True)
class _ShapeAndDtype:
    shape: tuple
    dtype: np.dtype


# Passthrough batcher for tf saved models. Assumes the first dimension is batched and no other.
def _tf_passthrough_batcher(
    fn: Callable,
    inp: JaxArrayOrMap,
    batched_args: tuple,
    batched_dims: tuple,
    callable_flat_tf: Callable[[list[TfVal]], Sequence[TfVal]],
    **_kwargs: dict,
) -> tuple:
    assert len(batched_dims) == 1
    assert len(batched_args) == 1
    treedef = jax.tree_structure(inp)
    # Map non-integers to 1 to handle polymorphic inputs, e.g. on save.
    input_shape = tuple([int(v) if isinstance(v, int) else 1 for v in batched_args[0].shape])
    # Force call callable_flat_tf to fill `res_treedef` inside it.
    out = callable_flat_tf(np.zeros(input_shape))
    assert len(out) == 1
    output_shape = out[0].shape
    # Grab value which may be non-integer (polymorphic) from the batch dimension.
    output_shape = (batched_args[0].shape[0], *list(output_shape[1:]))
    # Assumes a single output.
    output_shape_dtype = _ShapeAndDtype(shape=output_shape, dtype=np.float32)
    args = treedef.unflatten(batched_args)
    ret = jax2tf.call_tf(fn, output_shape_dtype=output_shape_dtype)(args)
    return ([ret], (0,))
