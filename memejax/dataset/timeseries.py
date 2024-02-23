from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd
from dataclasses_json import DataClassJsonMixin, Undefined, dataclass_json


@dataclass_json(undefined=Undefined.RAISE)
@dataclass(eq=True, kw_only=True, order=True)
class Window(DataClassJsonMixin):
    past_win: tuple[int, int] | None = None
    fut_win: tuple[int, int] | None = None


WindowSet = list[Window]


@dataclass_json(undefined=Undefined.RAISE)
@dataclass(eq=True, kw_only=True, order=True, frozen=True)
class TimeseriesKey(DataClassJsonMixin):
    past_cnt: int
    """How many past values to include in the data passed to the backtest strategy."""

    fut_cnt: int
    """How many future values to include in the data."""


def timeseries_window_sets(
    data: dict[Any, pd.DataFrame], keys: Sequence[TimeseriesKey]
) -> list[WindowSet]:
    """Computes list containing every rolling window of the data."""

    # idxs holds the index of the start of the window for each key. Define the
    # key's present range as the time range from idx[i] + key.past_cnt - 1 to
    # idx[i] + key.past_cnt. The data at idx[i] + key.past_cnt - 1 will be the
    # last thing in the past window. Everything after that goes into the future
    # window. Both together is just called the window.
    idxs = [0 for _ in range(len(keys))]

    # Cur time denotes the time at the middle of the window - the present
    # (splitting past and future).
    cur_ts = max(df.index.min() for df in data.values())

    # Shape: (num samples, num keys, past/future (2), start/end index (2))
    window_sets: list[WindowSet] = []

    # Create numpy arrays of the indexes for faster access
    df_indexes = {k: df.index.to_numpy() for k, df in data.items()}

    done = False
    while not done:
        # Scan through for each key and find all combinations of indices
        # where each key's present range contains cur_time, and add those
        # windows as a sample. Then advance the cur_time.
        found_window_set = True
        for i, key in enumerate(keys):
            df_index = df_indexes[key]

            # Advance the window of this key so its present range contains cur_time.
            # Exit if we run out of data.
            key_cnt = key.past_cnt + key.fut_cnt
            while idxs[i] + key_cnt < len(df_index) and df_index[idxs[i] + key.past_cnt] <= cur_ts:
                idxs[i] += 1

            # If this window starts after the current time, advance the
            # current time. This may happen if past_cnt is very large, for
            # example.
            if cur_ts < df_index[idxs[i] + key.past_cnt - 1]:
                cur_ts = df_index[idxs[i] + key.past_cnt - 1]
                found_window_set = False

        if found_window_set:
            win_set: WindowSet = []
            for i, key in enumerate(keys):
                win: Window = Window()
                if key.past_cnt > 0:
                    win.past_win = (idxs[i], idxs[i] + key.past_cnt)
                if key.fut_cnt > 0:
                    win.fut_win = (idxs[i] + key.past_cnt, idxs[i] + key.past_cnt + key.fut_cnt)
                win_set.append(win)
            window_sets.append(win_set)
            # We added a window, advance the current time by the minimum
            # possible to require advancing a key's current index. Choose
            # the earliest next time from all keys. If the future window is
            # size zero, we may end here if all past windows are at the end
            # of each key's data.
            next_ts = None
            for i, key in enumerate(keys):
                df_index = df_indexes[key]
                idx = idxs[i] + key.past_cnt
                if idx < len(df_index) and (next_ts is None or df_index[idx] < next_ts):
                    next_ts = df_index[idx]
            # We are done if all keys are at the end of their data
            if next_ts is None or next_ts <= cur_ts:
                done = True
            else:
                cur_ts = next_ts
    print(f"Computed {len(window_sets)} data windows")
    return window_sets


def resample_timeseries_dict(dfs: dict[Any, pd.DataFrame]) -> dict[Any, pd.DataFrame]:
    """Resamples all dataframes in the dict to the union of all indexes."""
    # Concatenate all dataframes
    combined_df = pd.concat(dfs.values(), axis=1)
    combined_index = combined_df.index.unique().sort_values()

    # Reindex with the index in order.
    output_dfs = {}
    for key, df in dfs.items():
        output_df = df.reindex(combined_index)
        # Can't use interpolation, must use ffill to avoid leaking future data.
        # If use use e.g. time interpolation, taking a window of the data will
        # leak future data based on the trajectory of the interpolation.
        output_df = output_df.interpolate(method="ffill")
        # For the very beginning of the data, use 0.0 for missing values.
        output_df = output_df.fillna(0.0)
        output_dfs[key] = output_df

    return output_dfs


def timestamp_one_hot(parts: list[tuple[int, int]]) -> list[float]:
    onehot = []
    for value, size in parts:
        onehot.extend([1.0 if i == value else 0.0 for i in range(size)])
    return onehot


def mdwtime_one_hot(ts: pd.Timestamp) -> list[float]:
    parts = [
        (ts.month - 1, 12),
        (ts.day - 1, 31),
        (ts.weekday(), 7),
        (ts.hour, 24),
        (ts.minute, 60),
    ]
    return timestamp_one_hot(parts)


def dwtime_one_hot(ts: pd.Timestamp) -> list[float]:
    parts = [(ts.day - 1, 31), (ts.weekday(), 7), (ts.hour, 24), (ts.minute, 60)]
    return timestamp_one_hot(parts)


def wtime_one_hot(ts: pd.Timestamp) -> list[float]:
    parts = [(ts.weekday(), 7), (ts.hour, 24), (ts.minute, 60)]
    return timestamp_one_hot(parts)


def time_one_hot(ts: pd.Timestamp) -> list[float]:
    parts = [(ts.hour, 24), (ts.minute, 60)]
    return timestamp_one_hot(parts)
