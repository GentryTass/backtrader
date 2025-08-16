"""Starter kit for OI clustering and GEX based strategies.

This module implements the core calculations used by two option
strategies described in the accompanying specification:

* Pin Reversion – mean reversion around dominant open interest strikes
* Short Gamma Momentum – trend following during negative dealer gamma regimes

The functions expect pre–aggregated option data for a given underlying
and expiry.  They purposely avoid any I/O so that the user can wire the
logic to custom data sources or a backtesting engine such as
``backtrader``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, sqrt, pi
from typing import Iterable, List, Optional

import numpy as np


SQRT_2PI = sqrt(2 * pi)


def _norm_pdf(x: np.ndarray) -> np.ndarray:
    """Standard normal probability density function."""
    return np.exp(-0.5 * x * x) / SQRT_2PI


def black_scholes_gamma(
    s: float,
    k: np.ndarray,
    t: float,
    r: float,
    q: float,
    sigma: np.ndarray,
) -> np.ndarray:
    """Black–Scholes gamma for calls/puts.

    Parameters
    ----------
    s : float
        Underlying spot price.
    k : array_like
        Strike prices.
    t : float
        Time to expiry in years.
    r : float
        Risk free rate.
    q : float
        Dividend yield.
    sigma : array_like
        Implied volatilities for the options.
    """
    if t <= 0:
        raise ValueError("time to expiry must be positive")
    sigma = np.asarray(sigma, dtype=float)
    k = np.asarray(k, dtype=float)
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    return np.exp(-q * t) * _norm_pdf(d1) / (s * sigma * np.sqrt(t))


@dataclass
class OptionMetrics:
    """Container with option positioning metrics."""

    dominant_strike: float
    concentration: float
    distance: float
    gex: float
    dominant_ratio: float


def compute_metrics(
    s: float,
    strikes: Iterable[float],
    call_oi: Iterable[float],
    put_oi: Iterable[float],
    call_iv: Iterable[float],
    put_iv: Iterable[float],
    t: float,
    r: float = 0.0,
    q: float = 0.0,
) -> OptionMetrics:
    """Compute dominant strike, concentration and GEX proxy."""

    strikes = np.asarray(list(strikes), dtype=float)
    call_oi = np.asarray(list(call_oi), dtype=float)
    put_oi = np.asarray(list(put_oi), dtype=float)
    call_iv = np.asarray(list(call_iv), dtype=float)
    put_iv = np.asarray(list(put_iv), dtype=float)

    coi = call_oi + put_oi
    order = np.argsort(coi)[::-1]
    top3 = coi[order[:3]]
    dominant = order[0]
    dominant_strike = strikes[dominant]
    concentration = top3[0] / np.sum(top3)
    dominant_ratio = top3[0] / top3[1] if top3.size > 1 and top3[1] > 0 else np.inf
    distance = abs(s - dominant_strike) / s

    gamma_call = black_scholes_gamma(s, strikes, t, r, q, call_iv)
    gamma_put = black_scholes_gamma(s, strikes, t, r, q, put_iv)
    gex = np.sum(
        gamma_call * call_oi * 100 * s * s - gamma_put * put_oi * 100 * s * s
    )

    return OptionMetrics(
        dominant_strike=dominant_strike,
        concentration=float(concentration),
        distance=float(distance),
        gex=float(gex),
        dominant_ratio=float(dominant_ratio),
    )


def pin_reversion_signal(metrics: OptionMetrics, has_catalyst: bool = False) -> bool:
    """Check entry conditions for the pin reversion setup."""

    return (
        metrics.concentration >= 0.50
        and metrics.dominant_ratio >= 1.5
        and metrics.distance <= 0.02
        and metrics.gex > 0
        and not has_catalyst
    )


def short_gamma_momentum_signal(
    metrics: OptionMetrics,
    gex_history: Iterable[float],
    prices: Iterable[float],
) -> Optional[str]:
    """Return momentum trade direction based on negative GEX.

    Parameters
    ----------
    metrics : OptionMetrics
        Current option metrics.
    gex_history : iterable
        Historical GEX values (must include the most recent ones prior to
        ``metrics.gex``).
    prices : iterable
        Recent closing prices including the current close.

    Returns
    -------
    str or None
        ``"call"`` for long call vertical, ``"put"`` for long put vertical
        or ``None`` when no trade is suggested.
    """
    gex_hist = np.asarray(list(gex_history), dtype=float)
    if gex_hist.size < 60:
        raise ValueError("gex_history requires at least 60 observations")

    threshold = np.percentile(np.abs(gex_hist), 60)
    prices = np.asarray(list(prices), dtype=float)
    if prices.size < 3:
        raise ValueError("prices requires at least 3 observations")
    prev_high = prices[:-1][-2:].max()
    prev_low = prices[:-1][-2:].min()
    close = prices[-1]

    if (
        metrics.gex < 0
        and abs(metrics.gex) > threshold
        and not (
            metrics.concentration >= 0.50 and metrics.distance <= 0.015
        )
    ):
        if close > prev_high:
            return "call"
        if close < prev_low:
            return "put"
    return None


if __name__ == "__main__":
    # rudimentary example with made-up numbers
    s = 100.0
    strikes = [95, 100, 105]
    call_oi = [500, 2000, 400]
    put_oi = [600, 1500, 300]
    call_iv = [0.2, 0.18, 0.19]
    put_iv = [0.22, 0.21, 0.2]
    t = 1 / 52  # one week

    metrics = compute_metrics(s, strikes, call_oi, put_oi, call_iv, put_iv, t)
    print(metrics)
    print("Pin reversion?", pin_reversion_signal(metrics))
    gex_hist = np.linspace(-1e8, 1e8, 60)
    prices = [97, 98, 102]
    print("Momentum direction:", short_gamma_momentum_signal(metrics, gex_hist, prices))
