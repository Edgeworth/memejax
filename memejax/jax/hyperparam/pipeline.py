from typing import Any

import flax.linen as nn
import jax
import optuna

from memejax.jax.pipeline.dataset import JaxDataset, JaxTrainData
from memejax.jax.pipeline.metrics import DeviceMetricDict, MetricsFn, get_local_mdict
from memejax.jax.pipeline.train_cfg import JaxTrainCfg
from memejax.jax.pipeline.trainer import JaxTrainer, JaxTrainMeta
from memejax.jax.util import JaxModelOutput


class JaxHyperParameterPipeline:
    cfg: JaxTrainCfg
    prefix: str
    train_ds: JaxDataset
    valid_ds: JaxDataset
    trainers: list[JaxTrainer]
    trial: optuna.Trial

    def __init__(
        self,
        *,
        prefix: str,
        trial: optuna.Trial,
        cfg: JaxTrainCfg,
        train_ds: JaxDataset,
        valid_ds: JaxDataset,
        model_cls: type[nn.Module],
        model_args: Any,
        metrics_fn: MetricsFn,
        sample_size: int = 1,
    ) -> None:
        self.prefix = prefix
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.cfg = cfg
        self.trial = trial
        self.rng = jax.random.PRNGKey(0)

        self.rng, rng = jax.random.split(self.rng)
        batched_sample = train_ds.batch(1, rng)

        self.trainers = []
        for _ in range(sample_size):
            self.rng, rng = jax.random.split(self.rng)
            trainer = JaxTrainer(
                rng=rng,
                cfg=cfg,
                batched_inp=batched_sample.model_inp,
                model_cls=model_cls,
                model_args=model_args,
                metrics_fn=metrics_fn,
            )
            self.trainers.append(trainer)

    def _train_epoch(
        self, trainer: JaxTrainer, meta: JaxTrainMeta
    ) -> tuple[DeviceMetricDict, tuple[JaxTrainData, JaxModelOutput]]:
        self.rng, rng = jax.random.split(self.rng)
        batches = self.train_ds.batches(self.cfg.epoch_batches, self.cfg.batch_size, rng)
        return trainer.train_epoch(batches, meta)

    def _validate(
        self, trainer: JaxTrainer
    ) -> tuple[DeviceMetricDict, tuple[JaxTrainData, JaxModelOutput]]:
        self.rng, rng = jax.random.split(self.rng)
        batches = self.valid_ds.batches(self.cfg.valid_batches, self.cfg.batch_size, rng)
        return trainer.validate(batches)

    def _run_epoch(self, epoch: int, max_epochs: int) -> list[float]:
        losses = []
        for trainer in self.trainers:
            _, _ = self._train_epoch(
                trainer=trainer, meta=JaxTrainMeta.from_data(epoch, max_epochs)
            )
            valid_dmdict, _ = self._validate(trainer)

            valid_mdict = get_local_mdict(valid_dmdict)
            losses.append(valid_mdict["loss"])

            if trainer.grad_zero_count > 10:
                raise optuna.exceptions.TrialPruned
        return losses

    def run_epoch(self, epoch: int, max_epochs: int) -> float:
        losses = self._run_epoch(epoch, max_epochs)
        print(f"{self.prefix} epoch {epoch}/{max_epochs}: {losses}")
        loss = sum(losses) / len(losses)
        # Skip if we are pruning or if we are getting too many grad zeros.
        if self.trial.should_prune():
            raise optuna.exceptions.TrialPruned
        return loss

    def run(self, epochs: int) -> float:
        loss = None
        for i in range(epochs):
            loss = self.run_epoch(i, epochs)
            self.trial.report(loss, i)
        assert loss is not None
        return loss
