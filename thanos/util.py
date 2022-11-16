"""
    Copyright (C) 2022 ServiceNow Inc
    @Author Lorne Schell <orange.toaster at gmail dot com>
    Some utility functions for stream manipulation and plotting of results.
"""

import itertools
from copy import deepcopy

import numpy as np
from thanos.gen import *

rng_fixed_singleton = np.random.default_rng()

"""
This returns a seeded numpy generator for use in a stream component.
"""


def seeded_gen(seedval: int):
    return np.random.Generator(np.random.PCG64(seedval))


def this_week():
    import pandas as pd

    return pd.date_range(
        end=pd.Timestamp.now().round("min"), freq="1min", periods=7 * 24 * 60
    )


def today_offset():
    import pandas as pd

    now = pd.Timestamp.now()
    return (now.hour - 12) * 60 + now.minute


def plot_examples(stream: Variable, example_count: int = 1):
    import pandas as pd
    from thanos.rebasis import ReBasis

    anomalies = []
    index = this_week()
    index_len = len(index)

    if example_count > 1:
        stream = ReBasis(deepcopy(rng_fixed_singleton), np.identity(example_count), [stream])
    elif hasattr(stream, "numdims"):
        example_count = stream.numdims

    fig = pd.DataFrame(data=fromiter(stream, float, count=index_len), index=index,).plot(
        marker=".",
        alpha=0.5,
        linestyle="none",
        figsize=(32, 3 * example_count),
        subplots=True,
    )
    # fig, ax = plt.subplots()
    for f, anom in zip(fig, anomalies):
        if anom is not None:
            for an in anom.values():
                for a in an:
                    if type(a) is tuple and a[1] < index_len:
                        f.axvspan(index[a[0]], index[a[1]], alpha=0.3, color="red")


def animate_plot(ds, batch_size=40, batch_count=10, subplots=False, figsize=(20, 5)):
    from IPython.display import clear_output
    from time import sleep
    import matplotlib
    from matplotlib.pyplot import close
    import torch
    import pandas as pd

    fig = matplotlib.pyplot.figure()
    hfig = display(fig, display_id=True)
    axes = fig.add_subplot(111)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=0)

    batches = None
    for (index, batch) in zip(range(batch_count), dl):
        if batches is None:
            batches = np.empty((0,) + batch.shape[1:])
        axes.cla()
        # clear_output(wait=True) - for non-pyplot output.
        batches = np.append(batches, batch.numpy(), axis=0)
        plt = pd.DataFrame(np.squeeze(batches)).plot(
            ax=axes,
            marker=".",
            alpha=0.5,
            linestyle="none",
            figsize=figsize,
            subplots=subplots,
        )
        # print(dl.dataset._buffer_size)
        # print(dir(ds))
        anomalies = []
        anomaly_set = set()
        anomalies.append(deepcopy(ds.anomalies()))
        for anomaly in anomalies:
            for k, v in anomaly.items():
                for start, end in v:
                    anomaly_set.add((start, end))
        # print(anomaly_set)
        for start, end in anomaly_set:
            if start < batches.shape[0]:
                plt.axvspan(start, end, alpha=0.3, color="red")
        fig.canvas.draw()
        hfig.update(fig)
        sleep(1)

    close()


def print_optree(root: Variable):
    def visitor(self, depth, parent):
        print("\t" * depth + self.__class__.__name__ + " " + str(self))

    root._recurse_helper(root, visitor)


""""
    The numpy implementation of fromiter doesn't support multidimensionality
"""


def fromiter(iterable: typing.Iterable[typing.Any], dtype: np.dtype, count: int):
    peek = next(iter(iterable))
    if type(peek) is np.ndarray:
        array = np.ndarray(shape=(count,) + peek.shape, dtype=dtype)
        for i, value in zip(range(count), itertools.chain([peek], iterable)):
            array[i, :] = value
        return array
    else:
        return np.fromiter(itertools.chain([peek], iterable), dtype, count)
