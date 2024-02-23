from collections.abc import Callable
from pathlib import Path
from typing import Any

import flax.linen as nn
import jax
import orbax.checkpoint as ocp
from jax import Array

from memejax.jax.jnp import ArrayMap, ModelOutput
from memejax.jax.pipeline.checkpoint import make_checkpoint_manager
from memejax.jax.pipeline.dataset import Dataset, TrainingData
from memejax.jax.pipeline.metrics import (
    DeviceMetricDict,
    LocalMetricDict,
    MetricsFn,
    get_local_mdict,
)
from memejax.jax.pipeline.reporter import Reporter
from memejax.jax.pipeline.train_cfg import TrainCfg
from memejax.jax.pipeline.trainer import Trainer, TrainMeta

ExampleFn = Callable[[tuple[TrainingData, ModelOutput], tuple[TrainingData, ModelOutput]], None]


class Pipeline:
    rng: Array
    cfg: TrainCfg
    train_ds: Dataset
    valid_ds: Dataset
    trainer: Trainer
    reporter: Reporter | None
    example_fn: ExampleFn | None
    ckpt_mgr: ocp.CheckpointManager | None = None
    best_ckpt_loss: float = float("inf")
    best_ckpt_epoch: int | None = None

    def __init__(
        self,
        *,
        rng: Array,
        cfg: TrainCfg,
        train_ds: Dataset,
        valid_ds: Dataset,
        model_cls: type[nn.Module],
        model_args: Any,
        metrics_fn: MetricsFn,
        example_fn: ExampleFn | None = None,
        report: bool = True,
    ) -> None:
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.cfg = cfg
        self.example_fn = example_fn

        self.rng, rng = jax.random.split(rng)
        batched_sample = train_ds.batch(1, rng)

        if report:
            self.reporter = Reporter(self.cfg)

        self.rng, rng = jax.random.split(rng)
        self.trainer = Trainer(
            rng=rng,
            cfg=cfg,
            batched_inp=batched_sample.model_inp,
            model_cls=model_cls,
            model_args=model_args,
            metrics_fn=metrics_fn,
            reporter=self.reporter,
        )

        if cfg.ckpt_max_secs:
            self.ckpt_mgr = make_checkpoint_manager(cfg.output_path, self.cfg.ckpt_max_secs)

    def _train_epoch(
        self, meta: TrainMeta
    ) -> tuple[DeviceMetricDict, tuple[TrainingData, ModelOutput]]:
        self.rng, rng = jax.random.split(self.rng)
        batches = self.train_ds.batches(self.cfg.epoch_batches, self.cfg.batch_size, rng)
        return self.trainer.train_epoch(batches, meta)

    def _validate(self) -> tuple[DeviceMetricDict, tuple[TrainingData, ModelOutput]]:
        self.rng, rng = jax.random.split(self.rng)
        batches = self.valid_ds.batches(self.cfg.valid_batches, self.cfg.batch_size, rng)
        return self.trainer.validate(batches)

    def _report(
        self,
        epoch_count: int,
        train_mdict: LocalMetricDict,
        valid_mdict: LocalMetricDict,
        done: bool,
    ) -> None:
        valid_loss = valid_mdict["loss"]

        if self.reporter:
            self.reporter.on_epoch(train_mdict, valid_mdict)

        # Call here to avoid overhead of checking every step (particularly with
        # converting metrics to local).
        if self.ckpt_mgr:
            # Force save if this is the last iteration.
            force_save = done
            # Force save if the best validation loss is 1% better.
            if self.best_ckpt_loss - valid_loss > 0.01 * abs(self.best_ckpt_loss):
                force_save = True
            saved = self.ckpt_mgr.save(
                epoch_count,
                args=ocp.args.StandardSave(self.trainer.save_ckpt()),
                metrics=valid_mdict,
                force=force_save,
            )
            self.ckpt_mgr.wait_until_finished()
            if saved and valid_loss < self.best_ckpt_loss:
                self.best_ckpt_loss = valid_loss
                self.best_ckpt_epoch = epoch_count
            if saved:
                print(f"SAVED CHECKPOINT! forced: {force_save}")
            print(f"  best saved loss: {self.best_ckpt_loss:.4f} on epoch {self.best_ckpt_epoch}")

    def train_epoch(self, epoch: int, max_epochs: int) -> DeviceMetricDict:
        train_dmdict, example_train = self._train_epoch(meta=TrainMeta.from_data(epoch, max_epochs))
        valid_dmdict, example_valid = self._validate()

        if self.reporter or self.ckpt_mgr:
            train_mdict = get_local_mdict(train_dmdict)
            valid_mdict = get_local_mdict(valid_dmdict)
            done = epoch == max_epochs - 1
            self._report(epoch, train_mdict, valid_mdict, done)

        if self.example_fn:
            self.example_fn(example_train, example_valid)

        return valid_dmdict

    def train(self, epochs: int) -> DeviceMetricDict:
        last_valid_dmdict = None
        for i in range(epochs):
            last_valid_dmdict = self.train_epoch(i, epochs)
        assert last_valid_dmdict
        return last_valid_dmdict

    def inference(self, batch_inp: ArrayMap) -> ModelOutput:
        return self.trainer.inference(batch_inp)

    def load_ckpt(self, path: Path, best: bool, step: int | None) -> None:
        self.trainer.load_ckpt(path, best, step)
