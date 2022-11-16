"""
    Copyright (C) 2022 ServiceNow Inc
    @Author Lorne Schell <orange.toaster at gmail dot com>
    SyntheticDataset is for interfacing a recipe to pytorch for direct use in
    model training and testing.
"""

import random
from copy import deepcopy
from typing import Dict

import numpy as np
import torch
import torch.utils.data as torchdata
from thanos.gen import Variable
from thanos.util import fromiter, seeded_gen



class SyntheticDataset(torchdata.IterableDataset):
    """
    We will save the RNG states so iterations are repeatable.
    Set a number of batches in your training loop, as we will continue to generate.
    """

    def __init__(
        self, signal: Variable, rng: np.random.Generator, buffer_size=1024
    ) -> None:
        self.index = 0
        self._buffer = None
        self._buffer_size = buffer_size
        self.signal = signal
        # self._resample() will be called on the first iteration

    def anomalies(self):
        result = self.signal._recurse_helper(self.signal, "get_event_indexes")
        return result

    def _resample(self):
        self._buffer = fromiter(self.signal, float, count=self._buffer_size)

    def __iter__(self):
        return self

    def __next__(self):
        i = self.index % self._buffer_size

        if i == 0 or self._buffer is None:
            self._resample()

        self.index += 1
        return self._buffer[i]
