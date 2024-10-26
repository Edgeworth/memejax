import functools
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast, override

import flax.linen as nn
import jax
import numpy as np
import optuna
from dataclasses_json import DataClassJsonMixin, Exclude, Undefined, config, dataclass_json
from jax.experimental import jax2tf
from jax.experimental.jax2tf.call_tf import TfVal, UnspecifiedOutputShapeDtype, call_tf_p
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

    def set_save(self, save: bool) -> None:
        self.sm.set_save(save)

    @override
    def optuna_params(
        self, trial: optuna.Trial, optuna_cfg: OptunaSearchCfg, prefix: str = ""
    ) -> "SavedModelBlkCfg":
        return self


class SavedModelBlk(nn.Module):
    blk_cfg: SavedModelBlkCfg

    @nn.compact
    def __call__[T: JaxArrayOrMap](self, inp: T, _train: bool) -> T:
        fn = self.blk_cfg.sm.tf_func()
        batching.primitive_batchers[call_tf_p] = functools.partial(_tf_passthrough_batcher, fn, inp)
        # call_tf_graph supports polymorphic inputs for saving to a SavedModel.
        # But, it does not work for training / running at all, just for saving.
        save = self.blk_cfg.sm.save
        output = {self.blk_cfg.output_key: jax2tf.call_tf(fn, call_tf_graph=save)(inp)}
        return cast(T, output)


# Passthrough batcher for tf saved models. Assumes the first dimension is batched and no other.
def _tf_passthrough_batcher(
    fn: Callable,
    inp: JaxArrayOrMap,
    batched_args: tuple,
    batched_dims: tuple,
    call_tf_graph: bool,
    callable_flat_tf: Callable[[list[TfVal]], Sequence[TfVal]],
    **_kwargs: dict,
) -> tuple:
    assert len(batched_dims) == 1
    assert len(batched_args) == 1
    treedef = jax.tree.structure(inp)
    # Map non-integers to 1 to handle polymorphic inputs, e.g. on save.
    input_shape = tuple([int(v) if isinstance(v, int) else 1 for v in batched_args[0].shape])
    # Force call callable_flat_tf to fill `res_treedef` inside it.
    out = callable_flat_tf(np.zeros(input_shape))  # type: ignore[arg-type]
    assert len(out) == 1
    output_shape: list[int] = list(out[0].shape)
    # Grab value which may be non-integer (polymorphic) from the batch dimension.
    batched_output_shape = (batched_args[0].shape[0], *output_shape[1:])
    # Assumes a single output.
    output_shape_dtype = jax.ShapeDtypeStruct(shape=batched_output_shape, dtype=np.float32)
    args = jax.tree.unflatten(treedef, batched_args)
    ret = jax2tf.call_tf(
        fn,
        call_tf_graph=call_tf_graph,
        output_shape_dtype=cast(UnspecifiedOutputShapeDtype, output_shape_dtype),
    )(args)
    return ([ret], (0,))
