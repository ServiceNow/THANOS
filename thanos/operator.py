"""
    Copyright (C) 2022 ServiceNow Inc
    @Author Lorne Schell <orange.toaster at gmail dot com>
    These are some handy operators that aren't already in numpy.
"""

def sum(*params):
    sum = 0.0
    for p in params:
        sum += p
    return sum


def product(*params):
    multiplicand = 1.0
    for p in params:
        multiplicand *= p
    return multiplicand


def mix(*params):
    sum = 0.0
    multiplicand = 1.0
    for p in params:
        sum += p
        multiplicand *= p

    return 0.5 * sum + 0.5 * multiplicand
