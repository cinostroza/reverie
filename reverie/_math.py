"""Small numerics needed by the attribution engine.

The core library is dependency-free (HLD 4.7), so the Beta quantile that drives
every ranking decision is implemented here rather than pulled from scipy. The
test suite checks these against ``scipy.stats.beta`` to 1e-9.
"""

from __future__ import annotations

import math

__all__ = ["betainc", "beta_ppf", "beta_mean", "utility_lcb", "clamp"]

_MAX_ITER = 300
_EPS = 3.0e-16
_TINY = 1.0e-300


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _TINY:
        d = _TINY
    d = 1.0 / d
    h = d

    for m in range(1, _MAX_ITER + 1):
        m2 = 2 * m

        # Even step.
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _TINY:
            d = _TINY
        c = 1.0 + aa / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        h *= d * c

        # Odd step.
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _TINY:
            d = _TINY
        c = 1.0 + aa / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        delta = d * c
        h *= delta

        if abs(delta - 1.0) < _EPS:
            break

    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if not (0.0 <= x <= 1.0):
        raise ValueError(f"x must be in [0, 1], got {x}")
    if a <= 0.0 or b <= 0.0:
        raise ValueError(f"a and b must be positive, got a={a}, b={b}")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0

    log_front = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    front = math.exp(log_front)

    # The continued fraction converges quickly only on one side of the mode;
    # use the symmetry I_x(a,b) = 1 - I_{1-x}(b,a) for the other side.
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def beta_ppf(q: float, a: float, b: float) -> float:
    """Inverse CDF (quantile) of the Beta distribution.

    Bisection on ``betainc``. Slower than a Newton solve but unconditionally
    stable, which matters more here: ``a`` and ``b`` are frequently near 1.0
    (an unattributed memory carries the Beta(1,1) prior) where Newton on the
    incomplete beta is badly behaved.
    """
    if not (0.0 <= q <= 1.0):
        raise ValueError(f"q must be in [0, 1], got {q}")
    if a <= 0.0 or b <= 0.0:
        raise ValueError(f"a and b must be positive, got a={a}, b={b}")
    if q == 0.0:
        return 0.0
    if q == 1.0:
        return 1.0

    lo, hi = 0.0, 1.0
    # ~52 halvings takes the bracket below double precision.
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:  # bracket collapsed
            break
        if betainc(a, b, mid) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def beta_mean(a: float, b: float) -> float:
    return a / (a + b)


def utility_lcb(alpha: float, beta: float, quantile: float = 0.25) -> float:
    """Lower confidence bound on a memory's success rate.

    HLD 9.2. Ranking on the LCB rather than the mean is what stops a memory
    that succeeded once (Beta(2,1), mean 0.67) from outranking one that
    succeeded 40 of 45 times (Beta(41,6), mean 0.87 but LCB far higher).
    """
    return beta_ppf(quantile, alpha, beta)
