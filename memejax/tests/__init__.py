from dataclasses import dataclass

import numpy as np
import pandas as pd

from memejax.data.timeseries import TimeseriesKey, Window, mdwtime_one_hot, timeseries_window_sets


@dataclass(eq=True, kw_only=True, order=True, frozen=True)
class _TimeseriesKey(TimeseriesKey):
    key: str


def test_mdwtime_one_hot() -> None:
    # Test January 1st, 12:00
    ts = pd.Timestamp(year=2023, month=1, day=1, hour=12, minute=0)
    assert (
        mdwtime_one_hot(ts)
        == [1.0]
        + [0.0] * 11
        + [1.0]
        + [0.0] * 30
        + [0.0] * 6
        + [1.0]
        + [0.0] * 12
        + [1.0]
        + [0.0] * 11
        + [1.0]
        + [0.0] * 59
    )

    # Test March 15th, 3:45
    ts = pd.Timestamp(year=2023, month=3, day=15, hour=3, minute=45)
    assert (
        mdwtime_one_hot(ts)
        == [0.0] * 2
        + [1.0]
        + [0.0] * 9
        + [0.0] * 14
        + [1.0]
        + [0.0] * 16
        + [0.0] * 2
        + [1.0]
        + [0.0] * 4
        + [0.0] * 3
        + [1.0]
        + [0.0] * 20
        + [0.0] * 45
        + [1.0]
        + [0.0] * 14
    )

    # Test December 31st, 23:59 (edge case)
    ts = pd.Timestamp(year=2023, month=12, day=31, hour=23, minute=59)
    assert mdwtime_one_hot(ts) == [0.0] * 11 + [1.0] + [0.0] * 30 + [1.0] + [0.0] * 6 + [1.0] + [
        0.0
    ] * 23 + [1.0] + [0] * 59 + [1.0]

    # Test February 29th, 0:00 (leap year)
    ts = pd.Timestamp(year=2024, month=2, day=29, hour=0, minute=0)
    assert (
        mdwtime_one_hot(ts)
        == [0.0]
        + [1.0]
        + [0.0] * 10
        + [0.0] * 28
        + [1.0]
        + [0.0] * 2
        + [0.0] * 3
        + [1.0]
        + [0.0] * 3
        + [1.0]
        + [0.0] * 23
        + [1.0]
        + [0.0] * 59
    )


def test_timeseries_window_sets_seeded_random() -> None:
    np.random.seed(42)  # noqa: NPY002
    n = 100

    daily_dates = pd.date_range(start="2023-01-01", end="2023-12-30", freq="D")
    daily_data = pd.DataFrame({"price": np.zeros(len(daily_dates))}, index=daily_dates)
    daily_data = daily_data.sample(n=n).sort_index()

    hourly_dates = pd.date_range(start="2023-01-01", end="2023-12-15", freq="H")
    hourly_data = pd.DataFrame({"price": np.zeros(len(hourly_dates))}, index=hourly_dates)
    hourly_data = hourly_data.sample(n=8 * n).sort_index()

    fut_cnt = 2
    keys = [
        _TimeseriesKey(key="daily", past_cnt=10, fut_cnt=fut_cnt),
        _TimeseriesKey(key="hourly", past_cnt=80, fut_cnt=fut_cnt),
    ]

    data = {keys[0]: daily_data, keys[1]: hourly_data}

    window_sets = timeseries_window_sets(data, keys)

    for window_set in window_sets:
        for i, spec in enumerate(keys):
            win = window_set[i]
            assert win.past_win is not None
            assert win.past_win[1] - win.past_win[0] == spec.past_cnt
            assert win.fut_win is not None
            assert win.fut_win[1] - win.fut_win[0] == fut_cnt


def test_timeseries_window_sets_data() -> None:
    index = pd.DatetimeIndex(
        [
            "2023-01-01",
            "2023-01-02",
            "2023-01-03",
            "2023-01-05",
            "2023-01-06",
            "2023-01-09",
            "2023-01-10",
            "2023-01-12",
            "2023-01-14",
            "2023-01-15",
        ]
    )
    daily_data = pd.DataFrame(index=index, data=[0] * len(index))

    index = pd.DatetimeIndex(
        [
            "2023-01-01 03:00:00",
            "2023-01-01 05:00:00",
            "2023-01-01 07:00:00",
            "2023-01-01 09:00:00",
            "2023-01-01 16:00:00",
            "2023-01-01 22:00:00",
            "2023-01-02 01:00:00",
            "2023-01-02 06:00:00",
            "2023-01-02 09:00:00",
            "2023-01-02 15:00:00",
            "2023-01-02 18:00:00",
            "2023-01-02 21:00:00",
            "2023-01-02 22:00:00",
            "2023-01-03 07:00:00",
            "2023-01-03 08:00:00",
            "2023-01-03 09:00:00",
            "2023-01-03 15:00:00",
            "2023-01-03 18:00:00",
            "2023-01-04 00:00:00",
            "2023-01-04 01:00:00",
            "2023-01-04 03:00:00",
            "2023-01-04 04:00:00",
            "2023-01-04 05:00:00",
            "2023-01-04 06:00:00",
            "2023-01-04 10:00:00",
            "2023-01-04 12:00:00",
            "2023-01-04 15:00:00",
            "2023-01-04 18:00:00",
            "2023-01-04 21:00:00",
            "2023-01-04 22:00:00",
            "2023-01-05 03:00:00",
            "2023-01-05 12:00:00",
            "2023-01-05 13:00:00",
            "2023-01-05 14:00:00",
            "2023-01-05 17:00:00",
            "2023-01-05 18:00:00",
            "2023-01-05 22:00:00",
            "2023-01-05 23:00:00",
            "2023-01-06 01:00:00",
            "2023-01-06 04:00:00",
            "2023-01-06 06:00:00",
            "2023-01-07 08:00:00",
            "2023-01-07 11:00:00",
            "2023-01-07 13:00:00",
            "2023-01-08 00:00:00",
            "2023-01-08 05:00:00",
            "2023-01-08 08:00:00",
            "2023-01-08 09:00:00",
            "2023-01-08 11:00:00",
            "2023-01-08 13:00:00",
            "2023-01-08 14:00:00",
            "2023-01-08 17:00:00",
            "2023-01-09 03:00:00",
            "2023-01-09 07:00:00",
            "2023-01-09 10:00:00",
            "2023-01-09 11:00:00",
            "2023-01-09 16:00:00",
            "2023-01-10 11:00:00",
            "2023-01-10 12:00:00",
            "2023-01-10 17:00:00",
            "2023-01-10 18:00:00",
            "2023-01-10 22:00:00",
            "2023-01-11 00:00:00",
            "2023-01-11 09:00:00",
            "2023-01-12 08:00:00",
            "2023-01-12 10:00:00",
            "2023-01-12 14:00:00",
            "2023-01-12 18:00:00",
            "2023-01-12 21:00:00",
            "2023-01-12 23:00:00",
            "2023-01-13 00:00:00",
            "2023-01-13 06:00:00",
            "2023-01-13 09:00:00",
            "2023-01-13 16:00:00",
            "2023-01-13 21:00:00",
            "2023-01-13 22:00:00",
            "2023-01-14 03:00:00",
            "2023-01-14 11:00:00",
            "2023-01-14 14:00:00",
            "2023-01-14 18:00:00",
        ]
    )
    hourly_data = pd.DataFrame(index=index, data=[0] * len(index))

    keys = [
        _TimeseriesKey(key="daily", past_cnt=4, fut_cnt=1),
        _TimeseriesKey(key="hourly", past_cnt=3, fut_cnt=0),
    ]
    data = {keys[0]: daily_data, keys[1]: hourly_data}

    window_sets = timeseries_window_sets(data, keys)

    expected_window_sets = [
        [Window(past_win=(0, 4), fut_win=(4, 5)), Window(past_win=(27, 30), fut_win=None)],
        [Window(past_win=(0, 4), fut_win=(4, 5)), Window(past_win=(28, 31), fut_win=None)],
        [Window(past_win=(0, 4), fut_win=(4, 5)), Window(past_win=(29, 32), fut_win=None)],
        [Window(past_win=(0, 4), fut_win=(4, 5)), Window(past_win=(30, 33), fut_win=None)],
        [Window(past_win=(0, 4), fut_win=(4, 5)), Window(past_win=(31, 34), fut_win=None)],
        [Window(past_win=(0, 4), fut_win=(4, 5)), Window(past_win=(32, 35), fut_win=None)],
        [Window(past_win=(0, 4), fut_win=(4, 5)), Window(past_win=(33, 36), fut_win=None)],
        [Window(past_win=(0, 4), fut_win=(4, 5)), Window(past_win=(34, 37), fut_win=None)],
        [Window(past_win=(0, 4), fut_win=(4, 5)), Window(past_win=(35, 38), fut_win=None)],
        [Window(past_win=(1, 5), fut_win=(5, 6)), Window(past_win=(35, 38), fut_win=None)],
        [Window(past_win=(1, 5), fut_win=(5, 6)), Window(past_win=(36, 39), fut_win=None)],
        [Window(past_win=(1, 5), fut_win=(5, 6)), Window(past_win=(37, 40), fut_win=None)],
        [Window(past_win=(1, 5), fut_win=(5, 6)), Window(past_win=(38, 41), fut_win=None)],
        [Window(past_win=(1, 5), fut_win=(5, 6)), Window(past_win=(39, 42), fut_win=None)],
        [Window(past_win=(1, 5), fut_win=(5, 6)), Window(past_win=(40, 43), fut_win=None)],
        [Window(past_win=(1, 5), fut_win=(5, 6)), Window(past_win=(41, 44), fut_win=None)],
        [Window(past_win=(1, 5), fut_win=(5, 6)), Window(past_win=(42, 45), fut_win=None)],
        [Window(past_win=(1, 5), fut_win=(5, 6)), Window(past_win=(43, 46), fut_win=None)],
        [Window(past_win=(1, 5), fut_win=(5, 6)), Window(past_win=(44, 47), fut_win=None)],
        [Window(past_win=(1, 5), fut_win=(5, 6)), Window(past_win=(45, 48), fut_win=None)],
        [Window(past_win=(1, 5), fut_win=(5, 6)), Window(past_win=(46, 49), fut_win=None)],
        [Window(past_win=(1, 5), fut_win=(5, 6)), Window(past_win=(47, 50), fut_win=None)],
        [Window(past_win=(1, 5), fut_win=(5, 6)), Window(past_win=(48, 51), fut_win=None)],
        [Window(past_win=(1, 5), fut_win=(5, 6)), Window(past_win=(49, 52), fut_win=None)],
        [Window(past_win=(2, 6), fut_win=(6, 7)), Window(past_win=(49, 52), fut_win=None)],
        [Window(past_win=(2, 6), fut_win=(6, 7)), Window(past_win=(50, 53), fut_win=None)],
        [Window(past_win=(2, 6), fut_win=(6, 7)), Window(past_win=(51, 54), fut_win=None)],
        [Window(past_win=(2, 6), fut_win=(6, 7)), Window(past_win=(52, 55), fut_win=None)],
        [Window(past_win=(2, 6), fut_win=(6, 7)), Window(past_win=(53, 56), fut_win=None)],
        [Window(past_win=(2, 6), fut_win=(6, 7)), Window(past_win=(54, 57), fut_win=None)],
        [Window(past_win=(3, 7), fut_win=(7, 8)), Window(past_win=(54, 57), fut_win=None)],
        [Window(past_win=(3, 7), fut_win=(7, 8)), Window(past_win=(55, 58), fut_win=None)],
        [Window(past_win=(3, 7), fut_win=(7, 8)), Window(past_win=(56, 59), fut_win=None)],
        [Window(past_win=(3, 7), fut_win=(7, 8)), Window(past_win=(57, 60), fut_win=None)],
        [Window(past_win=(3, 7), fut_win=(7, 8)), Window(past_win=(58, 61), fut_win=None)],
        [Window(past_win=(3, 7), fut_win=(7, 8)), Window(past_win=(59, 62), fut_win=None)],
        [Window(past_win=(3, 7), fut_win=(7, 8)), Window(past_win=(60, 63), fut_win=None)],
        [Window(past_win=(3, 7), fut_win=(7, 8)), Window(past_win=(61, 64), fut_win=None)],
        [Window(past_win=(4, 8), fut_win=(8, 9)), Window(past_win=(61, 64), fut_win=None)],
        [Window(past_win=(4, 8), fut_win=(8, 9)), Window(past_win=(62, 65), fut_win=None)],
        [Window(past_win=(4, 8), fut_win=(8, 9)), Window(past_win=(63, 66), fut_win=None)],
        [Window(past_win=(4, 8), fut_win=(8, 9)), Window(past_win=(64, 67), fut_win=None)],
        [Window(past_win=(4, 8), fut_win=(8, 9)), Window(past_win=(65, 68), fut_win=None)],
        [Window(past_win=(4, 8), fut_win=(8, 9)), Window(past_win=(66, 69), fut_win=None)],
        [Window(past_win=(4, 8), fut_win=(8, 9)), Window(past_win=(67, 70), fut_win=None)],
        [Window(past_win=(4, 8), fut_win=(8, 9)), Window(past_win=(68, 71), fut_win=None)],
        [Window(past_win=(4, 8), fut_win=(8, 9)), Window(past_win=(69, 72), fut_win=None)],
        [Window(past_win=(4, 8), fut_win=(8, 9)), Window(past_win=(70, 73), fut_win=None)],
        [Window(past_win=(4, 8), fut_win=(8, 9)), Window(past_win=(71, 74), fut_win=None)],
        [Window(past_win=(4, 8), fut_win=(8, 9)), Window(past_win=(72, 75), fut_win=None)],
        [Window(past_win=(4, 8), fut_win=(8, 9)), Window(past_win=(73, 76), fut_win=None)],
        [Window(past_win=(5, 9), fut_win=(9, 10)), Window(past_win=(73, 76), fut_win=None)],
        [Window(past_win=(5, 9), fut_win=(9, 10)), Window(past_win=(74, 77), fut_win=None)],
        [Window(past_win=(5, 9), fut_win=(9, 10)), Window(past_win=(75, 78), fut_win=None)],
        [Window(past_win=(5, 9), fut_win=(9, 10)), Window(past_win=(76, 79), fut_win=None)],
        [Window(past_win=(5, 9), fut_win=(9, 10)), Window(past_win=(77, 80), fut_win=None)],
        [Window(past_win=(5, 9), fut_win=(9, 10)), Window(past_win=(77, 80), fut_win=None)],
    ]

    assert window_sets == expected_window_sets
