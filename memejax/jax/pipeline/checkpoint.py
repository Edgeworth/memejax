import datetime
from pathlib import Path
from typing import Any

import orbax.checkpoint as ocp


def make_checkpoint_manager(
    path: Path, min_ckpt_time: float | None = None
) -> ocp.CheckpointManager:
    keep_time_interval = None
    if min_ckpt_time:
        keep_time_interval = datetime.timedelta(seconds=min_ckpt_time)
    SAVE_INTERVAL = 500
    options = ocp.CheckpointManagerOptions(
        save_interval_steps=SAVE_INTERVAL,  # Save every N epochs.
        max_to_keep=100,
        # Keep every 5N epochs as well.
        keep_period=5 * SAVE_INTERVAL,
        keep_time_interval=keep_time_interval,
        best_mode="min",
        step_format_fixed_length=10,
        best_fn=lambda mdict: mdict["loss"],
    )

    return ocp.CheckpointManager(
        path, options=options, item_handlers=ocp.StandardCheckpointHandler()
    )


def select_checkpoint(path: Path, best: bool, step: int | None = None) -> Any:
    print(f"Loading from checkpoint at {path}")
    mgr = make_checkpoint_manager(path)
    if step is None:
        step = mgr.best_step() if best else mgr.latest_step()

    print(
        f"Checkpoint available steps {mgr.all_steps()}, latest step "
        f"{mgr.latest_step()}, best step {mgr.best_step()}; using step {step}"
    )

    # TODO(2): Remove this hack once https://github.com/google/orbax/issues/648 etc is fixed.
    for sharding_file in path.glob("**/_sharding"):
        sharding_file.unlink()  # This deletes the file
        print(f"Deleted: {sharding_file}")

    return mgr.restore(step)
