from pathlib import Path
from typing import Any

import flax.linen as nn
import jax
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
        print(out)
        return out

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

    def save_export(self, path: Path, metadata: str) -> None:
        # TODO(-1): fix this hardcoding for output.
        jax_module = JaxModule(
            self.state,
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
            ),
            ServingConfig(
                "metadata",
                # requires exactly one input, so make one up here.
                tf_preprocessor=lambda *_: [],
                input_signature=[],
                method_key="metadata",
            ),
        ]
        export_mgr = ExportManager(jax_module, serving_configs=serving_configs)
        export_mgr.save(path)


class JaxSavedModelInference:
    model: Any

    def __init__(self, *, path: Path) -> None:
        print(tf.config.list_logical_devices())
        print(jax.default_backend().upper())
        self.model = tf.saved_model.load(path)

    def inference(self, batch_inp: JaxArrayMap) -> JaxModelOutput:
        return self.model.signatures["serving_default"](**batch_inp)["output_0"]

    def metadata(self) -> str:
        return "".join(chr(i) for i in self.model.signatures["metadata"]()["output_0"])
