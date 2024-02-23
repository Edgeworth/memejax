from clu import metric_writers, periodic_actions

from memejax.pipeline.metrics import (
    DeviceMetricDict,
    LocalMetricDict,
    get_local_mdict,
    jnp_add_dmdicts,
    pretty_print_mdict,
    slow_div_mdict,
)
from memejax.pipeline.train_cfg import TrainCfg


class Reporter:
    """Reports info to tensorboard, the console, etc."""

    cfg: TrainCfg
    writer: metric_writers.SummaryWriter
    periodic_progress: periodic_actions.ReportProgress
    periodic_cb: list[periodic_actions.PeriodicCallback]
    step: int = 0
    epoch: int = 0

    print_dmdict: DeviceMetricDict
    print_count: float = 0.0
    report_dmdict: DeviceMetricDict
    report_count: float = 0.0

    def __init__(self, cfg: TrainCfg) -> None:
        self.cfg = cfg
        self.print_dmdict = {}
        self.report_dmdict = {}
        self.writer = metric_writers.SummaryWriter(str(cfg.output_path / "tb"))

        self.periodic_progress = periodic_actions.ReportProgress(
            every_secs=cfg.report_secs, writer=self.writer
        )

        self.periodic_cb = [
            periodic_actions.PeriodicCallback(
                every_secs=cfg.report_secs,
                every_steps=cfg.report_batches,
                callback_fn=self._report_metrics,
            ),
            periodic_actions.PeriodicCallback(
                every_secs=cfg.print_secs,
                every_steps=cfg.print_batches,
                callback_fn=self._print_metrics,
            ),
            # TODO(1): save checkpoint regularly
        ]

    def _print_metrics(self, step: int, t: float) -> None:  # noqa: ARG002
        if len(self.print_dmdict) == 0:
            return
        print(f"batch {step}:")
        dmdict = slow_div_mdict(self.print_dmdict, self.print_count)
        pretty_print_mdict(get_local_mdict(dmdict))
        self.print_dmdict = {}
        self.print_count = 0.0

    def _report_metrics(self, step: int, t: float) -> None:  # noqa: ARG002
        # skip if nothing to report (may happen with timing based callbacks)
        if len(self.report_dmdict) == 0:
            return
        dmdict = slow_div_mdict(self.report_dmdict, self.report_count)
        self.writer.write_scalars(self.step, get_local_mdict(dmdict))
        self.report_dmdict = {}
        self.report_count = 0.0

    def on_step(self, dmdict: DeviceMetricDict) -> None:
        self.print_dmdict = jnp_add_dmdicts(self.print_dmdict, dmdict)
        self.print_count += 1.0
        self.report_dmdict = jnp_add_dmdicts(self.report_dmdict, dmdict)
        self.report_count += 1.0

        self.step += 1
        self.periodic_progress(self.step)
        for p in self.periodic_cb:
            p(self.step)

    def on_epoch(self, train_mdict: LocalMetricDict, valid_mdict: LocalMetricDict) -> None:
        # TODO(1): save checkpoint if has best validation loss
        self.epoch += 1

        # Also useful to report epoch training metrics, in case there are epoch
        # only metrics.
        self.writer.write_scalars(self.step, train_mdict)
        self.writer.write_scalars(self.step, {"validation/" + k: v for k, v in valid_mdict.items()})

        name = self.cfg.output_path.name
        print(f"Epoch {self.epoch} - train {name}:")
        pretty_print_mdict(train_mdict)
        print(" validation:")
        pretty_print_mdict(valid_mdict)
