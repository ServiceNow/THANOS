"""
    Copyright (C) 2022 ServiceNow Inc
    @Author Lorne Schell <orange.toaster at gmail dot com>
    Please help us build a nice signal library! Add your favourite recipes here
"""

def averaged_levelled_noise():
    return npfilt(
        op(
            operator.add,
            ConstantVariable(RandomVariable("uniform", seed(1), 0.0, 10.0)),
            RandomVariable("gamma", seed(1), 1.0, 2.0),
        ),
        7,
        np.convolve,
        [0.5, 0.5],
        mode="valid",
    )
