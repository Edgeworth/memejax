from pathlib import Path
from typing import Any, cast

import flax.linen as nn
import jax.numpy as jnp
import tensorflow as tf
from orbax.export import ExportManager, JaxModule, ServingConfig

from memejax.jax.pipeline.checkpoint import select_checkpoint
from memejax.jax.pipeline.trainer import JaxTrainer
from memejax.jax.util import JaxArrayMap, JaxModelOutput


class JaxCkptInference:
    model: nn.Module
    state: dict[str, Any]
    batched_inp: JaxArrayMap

    def __init__(
        self,
        *,
        batched_inp: JaxArrayMap,
        model_cls: type[nn.Module],
        model_args: Any,
        vmap_in: Any = None,
    ) -> None:
        self.model = JaxTrainer.vmap_model(model_cls, model_args, batched_inp, vmap_in)
        self.batched_inp = batched_inp

    @staticmethod
    def _apply_fn(
        model: nn.Module, state: dict[str, Any], batch_inp: JaxArrayMap
    ) -> JaxModelOutput:
        params = state["state"]["params"]
        batch_stats: JaxArrayMap = state["state"]["batch_stats"] or {}
        out = model.apply({"params": params, "batch_stats": batch_stats}, batch_inp, False)
        return out

    @staticmethod
    def _subset_state(state: dict[str, Any]) -> dict[str, Any]:
        # Record only the state we need for saving, because e.g. optax's optimisation state
        # uses some int32s which can't be saved for GPU models.
        return {
            "state": {
                "params": state["state"]["params"],
                "batch_stats": state["state"]["batch_stats"],
            }
        }

    def inference(self, batch_inp: JaxArrayMap) -> JaxModelOutput:
        return JaxCkptInference._apply_fn(self.model, self.state, batch_inp)

    def load_ckpt(self, path: Path, best: bool, step: int | None) -> None:
        self.state = select_checkpoint(path, best, step)

    def _array_map_to_input_signature(self, data: JaxArrayMap) -> Any:
        # Include a polymorphic batch dimension.
        data_signature = {
            k: tf.TensorSpec([None, *v.shape[1:]], v.dtype, name=k) for k, v in data.items()
        }
        return [data_signature]

    def save_export(
        self, path: Path, metadata: str, extra_trackable_resources: list | None = None
    ) -> None:
        # TODO(-1): fix this hardcoding for output.
        jax_module = JaxModule(
            self._subset_state(self.state),
            {
                "predict": lambda *args: JaxCkptInference._apply_fn(self.model, *args)["alloc_out"][
                    "alloc"
                ],
                "metadata": lambda *_: jnp.array([ord(i) for i in metadata]),
            },
            # Our jax code is polymorphic on the batch size, so let the export do that too.
            input_polymorphic_shape={"predict": "(b, ...)", "metadata": "(...)"},
        )

        serving_configs = [
            ServingConfig(
                "serving_default",
                input_signature=self._array_map_to_input_signature(self.batched_inp),
                method_key="predict",
                extra_trackable_resources=extra_trackable_resources,
            ),
            ServingConfig(
                "metadata",
                # requires exactly one input, so make one up here.
                tf_preprocessor=lambda *_: [],
                input_signature=[],
                method_key="metadata",
            ),
        ]
        if tf.executing_eagerly():
            export_mgr = ExportManager(jax_module, serving_configs=serving_configs)
            export_mgr.save(path)
        else:
            assert (
                extra_trackable_resources is not None
            ), "extra_trackable_resources required for graph mode"
            graph = (
                extra_trackable_resources[0].graph
                if extra_trackable_resources
                else tf.compat.v1.get_default_graph()
            )
            with tf.compat.v1.Session(graph=graph).as_default() as sess:
                # Run initializers
                if extra_trackable_resources:
                    sess.run([v.initializer for v in extra_trackable_resources])
                sess.run([v.initializer for v in jax_module.variables])

                export_mgr = ExportManager(jax_module, serving_configs=serving_configs)
                export_mgr.save(path)


class JaxSavedModelInference:
    model: Any
    save: bool = False

    def __init__(self, *, path: Path) -> None:
        self.model = tf.saved_model.load(path)

    def set_save(self, save: bool) -> None:
        # If we are saving this saved model, we need to use the graph mode
        # and call_tf_graph.
        self.save = save
        assert save != tf.executing_eagerly(), "for saving, disable eager execution"

    def inference(self, batch_inp: JaxArrayMap) -> JaxModelOutput:
        return self.model.signatures["serving_default"](**batch_inp)["output_0"]

    def tf_func(self) -> Any:
        return lambda inp: self.model.signatures["serving_default"](**inp)["output_0"]

    def metadata(self) -> str:
        return "".join(chr(i) for i in self.model.signatures["metadata"]()["output_0"])

    def extra_trackable_resources(self) -> list:
        variables = self.model.signatures["serving_default"].variables
        return cast(list, variables)

    def __deepcopy__(self, memo: dict[int, Any]) -> "JaxSavedModelInference":
        # Don't allow deep copy here - it breaks usage of this in json serialization.
        return self
