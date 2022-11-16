"""
    Copyright (C) 2022 ServiceNow Inc
    @Author Lorne Schell <orange.toaster at gmail dot com>
    This is the core file for Thanos Synthetic Timeseries Recipe Creation.
    It includes the class structure for streaming variables and random state
    tracking.
"""

import abc
import operator
import typing
from copy import deepcopy

import numpy as np


class Variable(metaclass=abc.ABCMeta):
    _is_variable_ = True

    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "__call__")
            and callable(subclass.__call__)
            and hasattr(subclass, "__iter__")
            and callable(subclass.__iter__)
        )

    @abc.abstractmethod
    def __call__(self):
        raise NotImplementedError

    def __iter__(self):
        while True:
            yield self()

    def apply(self, lambdop: callable, *params, **kwargs):
        return uniop(self, lambdop, *params, **kwargs)

    def window(self, window_size, sliding=False):
        return window(self, window_size, sliding)

    @classmethod
    def _recurse_helper(
        self,
        v,
        func: typing.Union[str, callable],
        *params,
        depth=0,
        prefix="",
        parent=None,
        **kwargs
    ):
        result = dict()
        if type(v) is dict:
            for subk, elem in v.items():
                result.update(
                    self._recurse_helper(
                        elem,
                        func,
                        *params,
                        depth=depth,
                        prefix=prefix + "." + str(subk),
                        parent=parent,
                        **kwargs
                    )
                )
        elif type(v) is tuple or type(v) is list:
            for i, elem in enumerate(v):
                result.update(
                    self._recurse_helper(
                        elem,
                        func,
                        *params,
                        depth=depth,
                        prefix=prefix + "." + str(i),
                        parent=parent,
                        **kwargs
                    )
                )
        elif callable(func) and hasattr(v, "_is_variable_"):
            res = func(v, *params, depth=depth, parent=parent, **kwargs)
            if res:
                result.update({prefix + str(v): res})
            result.update(
                self._recurse_helper(
                    vars(v),
                    func=func,
                    *params,
                    depth=depth + 1,
                    prefix=prefix + str(v),
                    parent=self,
                    **kwargs
                )
            )
        # isinstance doesn't work because it returns true for all collection types ??
        # type(v) is not str and type(v) is not np.ndarray and isinstance(v, Variable):
        elif hasattr(v, "_is_variable_"):
            if hasattr(v, func):
                res = getattr(v, func)(
                    *params,
                    depth=depth + 1,
                    prefix=prefix + str(v),
                    parent=self,
                    **kwargs
                )
                if res:
                    result.update({prefix + str(v): res})
            result.update(
                self._recurse_helper(
                    vars(v),
                    func,
                    *params,
                    depth=depth + 1,
                    prefix=prefix + str(v),
                    parent=self,
                    **kwargs
                )
            )
        return result

    def reset(self, *params, **kwargs):
        self._recurse_helper(vars(self), "reset")
        return self

    def reseed(self, rng: np.random.Generator, **kwargs):
        self._recurse_helper(vars(self), "reset")
        self._recurse_helper(vars(self), "reseed", rng)
        return self

    def __add__(self, other):
        if not callable(other):
            other = ConstantVariable(other)
        return op(operator.add, self, other)

    def __radd__(self, other):
        if not callable(other):
            other = ConstantVariable(other)
        return op(operator.add, other, self)

    def __sub__(self, other):
        if not callable(other):
            other = ConstantVariable(other)
        return op(operator.sub, self, other)

    def __rsub__(self, other):
        if not callable(other):
            other = ConstantVariable(other)
        return op(operator.sub, other, self)

    def __mod__(self, other):
        if not callable(other):
            other = ConstantVariable(other)
        return op(operator.mod, self, other)

    def __rmod__(self, other):
        if not callable(other):
            other = ConstantVariable(other)
        return op(operator.mod, other, self)

    def __mul__(self, other):
        if not callable(other):
            other = ConstantVariable(other)
        return op(operator.mul, self, other)

    def __rmul__(self, other):
        if not callable(other):
            other = ConstantVariable(other)
        return op(operator.mul, other, self)

    def __truediv__(self, other):
        if not callable(other):
            other = ConstantVariable(other)
        return op(operator.truediv, self, other)

    def __rtruediv__(self, other):
        if not callable(other):
            other = ConstantVariable(other)
        return op(operator.itruediv, other, self)

    def __pow__(self, other):
        if not callable(other):
            other = ConstantVariable(other)
        return op(operator.pow, self, other)

    def __rpow__(self, other):
        if not callable(other):
            other = ConstantVariable(other)
        return op(operator.pow, other, self)

    def __and__(self, other):
        if not callable(other):
            other = ConstantVariable(other)
        return op(operator.and_, self, other)

    def __rand__(self, other):
        if not callable(other):
            other = ConstantVariable(other)
        return op(operator.and_, other, self)

    def __or__(self, other):
        if not callable(other):
            other = ConstantVariable(other)
        return op(operator.or_, self, other)

    def __ror__(self, other):
        if not callable(other):
            other = ConstantVariable(other)
        return op(operator.or_, other, self)

    def __xor__(self, other):
        if not callable(other):
            other = ConstantVariable(other)
        return op(operator.xor, self, other)

    def __rxor__(self, other):
        if not callable(other):
            other = ConstantVariable(other)
        return op(operator.xor, other, self)

    def __gt__(self, other):
        if not callable(other):
            other = ConstantVariable(other)
        return op(operator.gt, self, other)

    def __ge__(self, other):
        if not callable(other):
            other = ConstantVariable(other)
        return op(operator.ge, self, other)

    def __lt__(self, other):
        if not callable(other):
            other = ConstantVariable(other)
        return op(operator.lt, self, other)

    def __le__(self, other):
        if not callable(other):
            other = ConstantVariable(other)
        return op(operator.le, self, other)

    def __eq__(self, other):
        if not callable(other):
            other = ConstantVariable(other)
        return op(operator.eq, self, other)

    def __ne__(self, other):
        if not callable(other):
            other = ConstantVariable(other)
        return op(operator.ne, self, other)


class RandomVariable(Variable):
    def __init__(
        self,
        distribution: str,
        rng: np.random.Generator = np.random.Generator(np.random.PCG64()),
        *params
    ):
        self.rng = rng
        self.initial_state = self.rng.bit_generator.state.copy()
        self.dist_name = distribution
        self.dist = getattr(self.rng, distribution)
        self.params = params

    def __call__(self):
        return self.dist(*[p() if callable(p) else p for p in self.params])

    def reset(self, *params, **kwargs):
        self.rng.bit_generator.state = self.initial_state.copy()
        return super().reset()

    def reseed(self, rng: np.random.Generator, **kwargs):
        self.rng = rng
        self.initial_state = self.rng.bit_generator.state.copy()
        self.dist = getattr(self.rng, self.dist_name)
        return super().reseed(rng)

    def __str__(self):
        return self.dist_name + "(" + ",".join(str(p) for p in self.params) + ")"


class ConstantVariable(Variable):
    def __init__(self, value):
        if callable(value):
            self.generator = value
            self.value = self.generator()
        else:
            self.value = value

    def __call__(self):
        return self.value

    def reseed(self, rng: np.random.Generator, **kwargs):
        retval = super().reseed(rng)
        if hasattr(self, "generator"):
            self.value = self.generator()
        return retval

    def __str__(self):
        return str(self.value)


class uniop(Variable):
    """
    Apply an operation to a signal sample-wise
    """

    def __init__(self, signal: Variable, op: callable, *params, **kwargs):
        self.params = params
        self.kwargs = kwargs
        self.signal = signal
        self.op = op

    def __call__(self):
        return self.op(self.signal(), *self.params, **self.kwargs)

    def __str__(self):
        return self.op.__name__ + str(self.params)


class op(Variable):
    """
    Apply an operation between some Variables
    """

    def __init__(self, op: callable, *params):
        self.params = params
        self.op = op

    def __call__(self):
        return self.op(*[param() for param in self.params])

    def __str__(self):
        return self.op.__name__


class window(Variable):
    """
    Groups the data stream into numpy array windows
    """

    def __init__(self, signal: Variable, window_size, sliding=False):
        self.signal = signal
        self.window_size = window_size
        self.sliding = sliding
        if self.sliding:
            self.buf = None

    def __call__(self):
        if self.sliding:
            if self.buf is None:
                self.buf = np.fromiter(self.signal, float, count=self.window_size)
            else:
                self.buf[:-1] = self.buf[1:]
                self.buf[-1] = self.signal()
            return self.buf.copy()
        else:
            return np.fromiter(self.signal, float, count=self.window_size)

    def apply(self, op: callable, *params, **kwargs):
        return uniop(self, op, *params, **kwargs)

    def reset(self, *params, **kwargs):
        self.buf = None
        return super().reset()

    def reseed(self, *params, **kwargs):
        self.buf = None
        return super().reseed(*params)


class npfilt(Variable):
    """
    the op must take a np.array
    """

    def __init__(
        self, signal: Variable, window_size: int, op: callable, *params, **kwargs
    ):
        self.op = op
        self.window_size = window_size
        self.signal = signal
        self.params = params
        self.kwargs = kwargs
        self.buf = None
        self.index = 0

    def __call__(self):
        if self.buf is None or self.index >= len(self.buf):
            self.index = 0
            self.buf = self.op(
                np.fromiter(self.signal, float, count=self.window_size),
                *self.params,
                **self.kwargs
            )

        self.index += 1
        return self.buf[self.index - 1]

    def reset(self, *params, **kwargs):
        self.buf = None
        return super().reset()

    def reseed(self, *params, **kwargs):
        self.buf = None
        return super().reseed(*params)


class Conditional(Variable):
    def __init__(self, condition: Variable, signal: Variable, then: Variable = None):
        self.signal = signal
        self.condition = condition
        self.then = then

    def __call__(self):
        if self.condition():
            return self.signal()
        elif self.then is not None:
            return self.then()


class StateSelector(Variable):
    """
    The transition variable returns the number of steps until a state change and the selection variable
    indexes the states, eg RandomVariable('integers', 0, len(states))
    """

    def __init__(
        self,
        transition: Variable,
        selection: Variable,
        states: typing.Mapping[int, Variable],
    ):
        self.transition = transition
        self.selection = selection
        self.index = self.selection()
        self.changepoint = self.transition()
        self.states = states

    def __call__(self):
        if self.changepoint <= 0:
            self.changepoint = self.transition()
            self.index = self.selection()
        self.changepoint -= 1
        return self.states[self.index]()

    def reset(self, *params, **kwargs):
        self.index = 0
        return super().reset()


class Seasonality(Variable):
    def __init__(self, period: int = 60 * 24, phase: int = 0):
        self.period = period
        self.phase = phase
        self.index = -1

    def __call__(self):
        self.index += 1
        return np.sin(2.0 / self.period * np.pi * (self.index + self.phase))

    def reset(self, *params, **kwargs):
        self.index = -1
        return super().reset()


class Trend(Variable):
    def __init__(
        self,
        rate: Variable = 1 / (60 * 24),
        initial_value: Variable = 0.0,
        maximum: Variable = np.inf,
        minimum: Variable = 0.0,
    ):
        self.rate = rate
        if not callable(maximum):
            maximum = ConstantVariable(maximum)
        self.maximum = maximum
        if not callable(minimum):
            minimum = ConstantVariable(minimum)
        self.minimum = minimum
        if not callable(initial_value):
            initial_value = ConstantVariable(initial_value)
        self.initial_value = initial_value
        self.current_value = self.initial_value()

    def __call__(self):
        if (self.current_value > self.maximum)():
            self.current_value = self.minimum()
        self.current_value += self.rate
        if callable(self.current_value):
            return self.current_value()
        return self.current_value

    def reset(self, *params, **kwargs):
        self.current_value = deepcopy(self.initial_value)()
        return super().reset()


class DateModulation(Variable):
    import pandas as pd

    def __init__(self, dates: pd.DatetimeIndex, base_rate=0.0, workhours=1.0):
        self.dates = dates
        self.index = -1
        self.base_rate = base_rate
        self.workhours = workhours

    def __call__(self):
        self.index += 1
        if self.index > len(self.dates):
            raise StopIteration()
        else:
            dt = self.dates[self.index]
            return (
                self.base_rate
                + (dt.dayofweek < 5 and dt.hour > 9 and dt.hour < 18) * self.workhours
            )

    def reset(self, *params, **kwargs):
        self.index = -1
        return super().reset()

class RandomAnomaly(Variable):
    def __init__(
        self,
        distribution: str,
        rng: np.random.Generator = np.random.Generator(np.random.PCG64()),
        threshold: float = 1.0 - 1e-8,
        *params
    ):
        self.rng = rng
        self.initial_state = self.rng.bit_generator.state.copy()
        self.dist_name = distribution
        self.dist = getattr(self.rng, distribution)
        self.params = params
        self.index = 0
        self.indexes = []
        self.threshold = threshold
        self.state = 0

    def __call__(self):
        val = self.dist(*[p() if callable(p) else p for p in self.params])
        if val > self.threshold:
            if self.state:
                self.indexes.append((self.index, np.inf))
            elif len(self.indexes) > 0:
                self.indexes[-1] = (self.indexes[-1][0], self.index)

            self.state ^= 1

        self.index +=1
        return self.state

    def get_event_indexes(
        self, *params, **kwargs
    ) -> typing.Sequence[typing.Tuple[int, int]]:
        if self.indexes:
            return deepcopy(self.indexes)
        return None

    def reset(self, *params, **kwargs):
        self.rng.bit_generator.state = self.initial_state.copy()
        self.index = 0
        self.indexes = []
        return super().reset()

    def reseed(self, rng: np.random.Generator, **kwargs):
        self.rng = rng
        self.initial_state = self.rng.bit_generator.state.copy()
        self.dist = getattr(self.rng, self.dist_name)
        return super().reseed(rng)

    def __str__(self):
        return self.dist_name + "(" + ",".join(str(p) for p in self.params) + ")"

class BinMarkovAnomaly(Variable):
    def __init__(
        self, seed: np.random.Generator, p_on: float = 0.001, p_off: float = 0.05
    ):
        self.state = 0
        self.rng = seed
        self.initial_state = self.rng.bit_generator.state.copy()
        self.p_on = p_on
        self.p_off = p_off
        self.index = 0
        self.indexes = []

    def __call__(self):
        if self.state:
            self.state = 1 - self.rng.binomial(1, self.p_off)
            if not self.state and len(self.indexes) > 0:
                self.indexes[-1] = (self.indexes[-1][0], self.index)
        else:
            self.state = self.rng.binomial(1, self.p_on)
            if self.state:
                self.indexes.append((self.index, np.inf))

        self.index += 1
        return self.state

    def get_event_indexes(
        self, *params, **kwargs
    ) -> typing.Sequence[typing.Tuple[int, int]]:
        if self.indexes:
            return deepcopy(self.indexes)
        return None

    def reset(self, *params, **kwargs):
        self.rng.bit_generator.state = self.initial_state.copy()
        self.index = 0
        self.indexes = []
        return super().reset()

    def reseed(self, rng: np.random.Generator, **kwargs):
        # self.rng = rng
        # self.initial_state = self.rng.bit_generator.state.copy()
        return super().reseed(rng)

    def __str__(self):
        return (
            str(self.__class__.__name__)
            + "("
            + ",".join(str(p) for p in [self.p_on, self.p_off])
            + ")"
        )
