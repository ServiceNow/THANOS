"""
    Copyright (C) 2022 ServiceNow Inc
    @Author Lorne Schell <orange.toaster at gmail dot com>
    This object creates a vectorspace of a set of signals and can create a new
    basis space via matrix multiplication. This operates in a streaming manner.
"""

import numpy as np
import typing
from copy import deepcopy

from thanos.gen import Variable
from thanos.util import seeded_gen, fromiter


class ReBasis(Variable):
    def __init__(
        self,
        seed: np.random.Generator,
        kernel: np.array,
        vectors: typing.Sequence[Variable],
    ):
        self.rng = seed
        self.kernel = kernel
        if len(self.kernel.shape) == 2:
            self.numdims = self.kernel.shape[0]
            if self.numdims > 1:
                self._initial_seeds = [
                    seeded_gen(self.rng.integers(0, 2 ** 32))
                    for i in range(self.numdims)
                ]
                self.signals = []
                for i in range(self.numdims):
                    self.signals.append(
                        [
                            deepcopy(signal.reseed(deepcopy(self._initial_seeds[i])))
                            for signal in vectors
                        ]
                    )
        else:
            self.numdims = 1

        if self.numdims == 1:
            self.vectors = vectors

    def __call__(self):
        if self.numdims > 1:
            sample = np.array([[X() for X in signal] for signal in self.signals])
            return np.sum(self.kernel.T * sample.T, axis=0)
        else:
            return np.sum(self.kernel * np.array([X() for X in self.vectors]))

    def __str__(self):
        return (
            str(self.__class__.__name__)
            + "("
            + ",".join(str(p) for p in self.kernel)
            + ")"
        )

    def reset(self):
        if self.numdims > 1:
            print(self.signals)
            for seed, signal in zip(self._initial_seeds, self.signals):
                newseed = deepcopy(seed)
                for X in signal:
                    X.reseed(newseed)
        return self
