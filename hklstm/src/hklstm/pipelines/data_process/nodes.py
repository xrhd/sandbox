"""
This is a boilerplate pipeline 'data_process'
generated using Kedro 0.18.4
"""

import math
import warnings
from typing import Tuple, TypeVar

import numpy as np

# import pandas as pd
# import plotnine as gg

T = TypeVar("T")
Pair = Tuple[T, T]

# gg.theme_set(gg.theme_bw())
# warnings.filterwarnings("ignore")


def sine_seq(
    phase: float,
    seq_len: int,
    samples_per_cycle: int,
) -> Pair[np.ndarray]:
    """Returns x, y in [T, B] tensor."""
    t = np.arange(seq_len + 1) * (2 * math.pi / samples_per_cycle)
    t = t.reshape([-1, 1]) + phase
    sine_t = np.sin(t)
    return sine_t[:-1, :], sine_t[1:, :]


def generate_data(
    seq_len: int,
    train_size: int,
    valid_size: int,
) -> Pair[Pair[np.ndarray]]:
    phases = np.random.uniform(0.0, 2 * math.pi, [train_size + valid_size])
    all_x, all_y = sine_seq(phases, seq_len, 3 * seq_len / 4)

    all_x = np.expand_dims(all_x, -1)
    all_y = np.expand_dims(all_y, -1)
    train_x = all_x[:, :train_size]
    train_y = all_y[:, :train_size]

    valid_x = all_x[:, train_size:]
    valid_y = all_y[:, train_size:]

    return (train_x, train_y), (valid_x, valid_y)


class Dataset:
    """An iterator over a numpy array, revealing batch_size elements at a time."""

    def __init__(self, xy: Pair[np.ndarray], batch_size: int):
        self._x, self._y = xy
        self._batch_size = batch_size
        self._length = self._x.shape[1]
        self._idx = 0
        if self._length % batch_size != 0:
            msg = "dataset size {} must be divisible by batch_size {}."
            raise ValueError(msg.format(self._length, batch_size))

    def __next__(self) -> Pair[np.ndarray]:
        start = self._idx
        end = start + self._batch_size
        x, y = self._x[:, start:end], self._y[:, start:end]
        if end >= self._length:
            end = end % self._length
            assert end == 0  # Guaranteed by ctor assertion.
        self._idx = end
        return x, y


def data_process(train: Pair[np.ndarray], valid: Pair[np.ndarray], BATCH_SIZE: int):
    train_ds = Dataset(train, BATCH_SIZE)
    valid_ds = Dataset(valid, BATCH_SIZE)
    return train_ds, valid_ds
