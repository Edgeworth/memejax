import dataclasses
from collections.abc import Iterator, Mapping
from functools import partial
from pathlib import Path
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.core.scope import VariableDict
from flax.training import train_state
from jax import Array
from jax.typing import ArrayLike

from memejax.jax.pipeline.checkpoint import select_checkpoint
from memejax.jax.pipeline.dataset import JaxTrainData
from memejax.jax.pipeline.metrics import (
    DeviceMetricDict,
    MetricsFn,
    jnp_add_dmdicts,
    slow_div_mdict,
)
from memejax.jax.pipeline.reporter import Reporter
from memejax.jax.pipeline.train_cfg import JaxTrainCfg, OptimizerCfg, RegularizationKind
from memejax.jax.util import JaxArrayMap, JaxModelOutput, maybe_chexify


@struct.dataclass
class JaxTrainMeta(struct.PyTreeNode):
    # Denotes how far through the training we are: [0.0, 1.0].
    train_fraction: Array

    @staticmethod
    def _train_fraction(epoch: int, max_epochs: int) -> float:
        # Assumes epoch is 0-indexed.
        max_epochs = max_epochs - 1
        # Use some % of the epochs at the beginning and end to keep it stable.
        start_frac = 0.1
        end_frac = 0.2
        if epoch >= max_epochs * (1.0 - end_frac):
            return 1.0
        if epoch <= max_epochs * start_frac:
            return 0.0
        return (epoch - max_epochs * start_frac) / (max_epochs * (1.0 - start_frac - end_frac))

    @staticmethod
    def from_data(epoch: int, max_epochs: int) -> "JaxTrainMeta":
        return JaxTrainMeta(
            train_fraction=jnp.array(JaxTrainMeta._train_fraction(epoch, max_epochs))
        )


class JaxTrainState(train_state.TrainState):
    dropout_rng: Array = struct.field(pytree_node=True)
    batch_stats: JaxArrayMap = struct.field(pytree_node=True)
    meta: JaxTrainMeta = struct.field(pytree_node=True)

    def make_variables(self) -> dict[str, Mapping[str, Any]]:
        return {
            "params": self.params,
            "batch_stats": self.batch_stats,
            "meta": dataclasses.asdict(self.meta),
        }


class JaxTrainer:
    rng: Array
    cfg: JaxTrainCfg
    model: nn.Module
    state: JaxTrainState
    metrics_fn: MetricsFn
    reporter: Reporter | None

    check_count: int = 0
    report_epochs: int = 0
    grad_zero_count: int = 0

    @staticmethod
    def vmap_model(
        model_cls: type[nn.Module], model_args: Any, batched_inp: JaxArrayMap, vmap_in: Any = None
    ) -> nn.Module:
        # If there is no info on how to map each part of the input, assume it
        # should all be mapped over the first axis.
        if vmap_in is None:
            vmap_in = jax.tree_map(lambda _: 0, batched_inp)

        return nn.vmap(
            model_cls,
            # (map data according to vmap_in, don't map 'train' var)
            in_axes=(vmap_in, None),
            out_axes=0,
            # Keep variables the same between samples in the batch.
            variable_axes={"params": None, "batch_stats": None, "meta": None},
            # Dropout should have different rng for each sample.
            split_rngs={"params": False, "dropout": True},
            axis_name="batch",
        )(**model_args)

    def __init__(
        self,
        *,
        rng: Array,
        cfg: JaxTrainCfg,
        batched_inp: JaxArrayMap,
        model_cls: type[nn.Module],
        model_args: Any,
        metrics_fn: MetricsFn,
        vmap_in: Any = None,
        reporter: Reporter | None = None,
    ) -> None:
        self.cfg = cfg
        self.metrics_fn = metrics_fn
        self.reporter = reporter

        # Automatically batch the model. Don't vmap everything at the loss
        # function level since we need to specify how variables and rngs are
        # lifted.
        self.model = JaxTrainer.vmap_model(model_cls, model_args, batched_inp, vmap_in)

        chain = []
        if cfg.opt_cfg.adaptive_grad_clip is not None:
            chain.append(optax.adaptive_grad_clip(cfg.opt_cfg.adaptive_grad_clip))

        chain.extend([cfg.opt_cfg.get_optimizer()])
        tx = optax.chain(*chain)

        # Provide one sample to initialize the model.
        self.rng, rng = jax.random.split(rng)
        variables = self.model.init(rng, batched_inp, False)
        # params may not exist if there are no trainable parameters. This can happen if e.g. only
        # jax operations are used.
        params = variables.get("params", {})
        batch_stats = variables.get("batch_stats", {})

        param_count = jax.tree_util.tree_reduce(
            lambda x, y: jnp.add(x, y.size), params, jnp.array(0)
        )
        print(f"Training model with {param_count} parameters.")

        self.rng, rng = jax.random.split(rng)
        self.state = JaxTrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=tx,
            dropout_rng=rng,
            batch_stats=batch_stats,
            meta=JaxTrainMeta(train_fraction=jnp.array(0.0)),
        )

    @staticmethod
    @jax.jit
    def _l1_reg(x: ArrayLike, lmbda: ArrayLike) -> Array:
        return jnp.asarray(lmbda) * jnp.mean(jnp.abs(x))

    @staticmethod
    @jax.jit
    def _l2_reg(x: ArrayLike, lmbda: ArrayLike) -> Array:
        return jnp.asarray(lmbda) * jnp.mean(x**2.0)

    @staticmethod
    @jax.jit
    def _elastic_reg(x: ArrayLike, lmbda: ArrayLike, alpha: ArrayLike) -> Array:
        l1 = jnp.asarray(alpha * JaxTrainer._l1_reg(x, lmbda))
        l2 = jnp.asarray((1.0 - alpha) * JaxTrainer._l2_reg(x, lmbda))
        return l1 + l2

    @staticmethod
    @partial(jax.jit, static_argnames=("metrics_fn", "opt_cfg"))
    def _metrics(
        variables: VariableDict,
        output: JaxModelOutput,
        batch_aux: JaxArrayMap,
        metrics_fn: MetricsFn,
        opt_cfg: OptimizerCfg,
    ) -> DeviceMetricDict:
        # Do not map variables over the batch axis.
        metric_map = jax.vmap(metrics_fn, in_axes=(None, 0, 0), out_axes=0)(
            variables, output, batch_aux
        )
        dmdict: DeviceMetricDict = {}
        losses = []
        for name, mcol in metric_map.items():
            prev_len = len(dmdict)
            d = mcol.traced_mdict()
            losses.append(d["loss"])
            dmdict.update({f"{name}/{k}": v for k, v in d.items()})
            assert len(dmdict) == prev_len + len(d), f"Duplicate metric name in {d}"

        assert "loss" not in dmdict
        dmdict["loss"] = jnp.array(losses).sum()

        reg_fn = lambda x: JaxTrainer._l1_reg(x, lmbda=opt_cfg.reg_lambda)
        match opt_cfg.reg:
            case RegularizationKind.L2:
                reg_fn = lambda x: JaxTrainer._l2_reg(x, lmbda=opt_cfg.reg_lambda)
            case RegularizationKind.ELASTIC:
                reg_fn = lambda x: JaxTrainer._elastic_reg(
                    x, lmbda=opt_cfg.reg_lambda, alpha=opt_cfg.reg_alpha
                )

        if opt_cfg.reg != "none":
            reg_params = jax.tree_map(reg_fn, variables["params"])
            dmdict["loss"] += jnp.array(jax.tree_util.tree_leaves(reg_params)).sum()

        return dmdict

    @staticmethod
    @maybe_chexify
    @partial(jax.jit, static_argnames=("metrics_fn", "opt_cfg"))
    def _train_step(
        state: JaxTrainState, batch: JaxTrainData, metrics_fn: MetricsFn, opt_cfg: OptimizerCfg
    ) -> tuple[JaxTrainState, DeviceMetricDict, JaxModelOutput, JaxArrayMap]:
        dropout_rng = jax.random.fold_in(key=state.dropout_rng, data=state.step)

        def loss_fn(
            params: dict[str, Any],
        ) -> tuple[ArrayLike, tuple[DeviceMetricDict, JaxArrayMap, JaxArrayMap]]:
            variables = state.make_variables()
            # Use given params to make sure differentiation works.
            variables["params"] = params
            output, updates = state.apply_fn(
                variables,
                batch.model_inp,
                True,
                mutable=["batch_stats"],
                rngs={"dropout": dropout_rng},
            )
            mdict = JaxTrainer._metrics(variables, output, batch.aux, metrics_fn, opt_cfg)

            return mdict["loss"], (mdict, updates, output)

        (_, (dmdict, updates, output)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params
        )
        state = state.apply_gradients(grads=grads)
        state = state.replace(batch_stats=updates["batch_stats"])

        return state, dmdict, output, grads

    @staticmethod
    @jax.jit
    def _check_grads_zero(grads: JaxArrayMap) -> Array:
        return jax.tree_util.tree_reduce(
            lambda x, y: jnp.logical_and(x, jnp.allclose(y, 0.0)), grads, jnp.array(True)
        )

    @staticmethod
    @jax.jit
    def _check_param_range(state: JaxTrainState) -> tuple[Array, Array]:
        return jax.tree_util.tree_reduce(
            lambda x, y: (
                jax.lax.min(x[0], jnp.min(y.reshape(-1))),
                jax.lax.max(x[1], jnp.max(y.reshape(-1))),
            ),
            state.params,
            (jnp.array(jnp.inf), jnp.array(-jnp.inf)),
        )

    @staticmethod
    @jax.jit
    def _check_grad_range(grads: JaxArrayMap) -> tuple[Array, Array]:
        return jax.tree_util.tree_reduce(
            lambda x, y: (
                jax.lax.min(x[0], jnp.min(y.reshape(-1))),
                jax.lax.max(x[1], jnp.max(y.reshape(-1))),
            ),
            grads,
            (jnp.array(jnp.inf), jnp.array(-jnp.inf)),
        )

    def train_epoch(
        self, batches: Iterator[JaxTrainData], meta: JaxTrainMeta
    ) -> tuple[DeviceMetricDict, tuple[JaxTrainData, JaxModelOutput]]:
        # Keep things on the GPU for as long as possible.
        total_dmdict: DeviceMetricDict = {}
        last_output = None
        last_grads = None
        num_batches = 0.0

        self.state = self.state.replace(meta=meta)
        for batch in batches:
            self.state, dmdict, output, grads = self._train_step(
                self.state, batch, self.metrics_fn, self.cfg.opt_cfg
            )
            total_dmdict = jnp_add_dmdicts(total_dmdict, dmdict)
            num_batches += 1.0

            if self.reporter:
                self.reporter.on_step(dmdict)

            last_output = (batch, output)
            last_grads = grads

        assert last_output
        total_dmdict = slow_div_mdict(total_dmdict, num_batches)

        # TODO(0): move to reporter, output to tensorboard as well.
        # `last_grads` may be null if there are no trainable parameters.
        # This may happen e.g. training ensemble models where only jax operations are used to
        # combine the outputs.
        if last_grads and self.cfg.check_epochs and self.check_count % self.cfg.check_epochs == 0:
            if JaxTrainer._check_grads_zero(last_grads):
                self.grad_zero_count += 1
            else:
                self.grad_zero_count = 0

            if self.grad_zero_count > 5:
                print(f"GRADIENTS ARE ALL CLOSE TO ZERO {self.grad_zero_count} times")

        if self.cfg.report_epochs and self.report_epochs % self.cfg.report_epochs == 0:
            param_min, param_max = JaxTrainer._check_param_range(self.state)
            grad_min, grad_max = JaxTrainer._check_grad_range(last_grads)
            # TODO(0): add mean
            total_dmdict["param_min"] = param_min
            total_dmdict["param_max"] = param_max
            total_dmdict["grad_min"] = grad_min
            total_dmdict["grad_max"] = grad_max

        for k, v in vars(meta).items():
            total_dmdict[f"meta/{k}"] = v

        self.check_count += 1
        self.report_epochs += 1

        return total_dmdict, last_output

    @staticmethod
    @jax.jit
    def _apply_fn(state: JaxTrainState, batch_inp: JaxArrayMap) -> JaxModelOutput:
        return state.apply_fn(state.make_variables(), batch_inp, False)

    # @chex.chexify
    def validate(
        self, batches: Iterator[JaxTrainData]
    ) -> tuple[DeviceMetricDict, tuple[JaxTrainData, JaxModelOutput]]:
        # Compute metrics using validation data.
        total_dmdict: DeviceMetricDict = {}
        num_batches = 0.0
        last_output = None
        for batch in batches:
            output = JaxTrainer._apply_fn(self.state, batch.model_inp)
            dmdict = JaxTrainer._metrics(
                self.state.make_variables(), output, batch.aux, self.metrics_fn, self.cfg.opt_cfg
            )
            total_dmdict = jnp_add_dmdicts(total_dmdict, dmdict)
            num_batches += 1.0
            last_output = (batch, output)
        assert last_output
        total_dmdict = slow_div_mdict(total_dmdict, num_batches)
        return total_dmdict, last_output

    def inference(self, batch_inp: JaxArrayMap) -> JaxModelOutput:
        return JaxTrainer._apply_fn(self.state, batch_inp)

    def save_ckpt(self) -> dict:
        # Return a pytree of everything trainer needs to checkpoint.
        return {"state": self.state}

    def load_ckpt(self, path: Path, best: bool, step: int | None) -> None:
        # Load a pytree of everything trainer needs to checkpoint.
        state = select_checkpoint(path, best, step)
        self.state = state["state"]
