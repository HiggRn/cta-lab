import math

from common import Bar, Indicator


class OiFlow(Indicator):
    """EWMA of price direction confirmed by relative Open Interest change, in [-1, 1]."""

    def __init__(self, span: int = 10, baseline_span: int = 20):
        self._alpha = 2.0 / (span + 1.0)
        self._alpha_base = 2.0 / (baseline_span + 1.0)
        self._min_bars = baseline_span
        self._count = 0
        self._prev_close: float | None = None
        self._prev_oi: float | None = None
        self._ema_oi = 0.0
        self._value = 0.0

    @property
    def is_ready(self) -> bool:
        return self._count >= self._min_bars

    @property
    def value(self) -> float:
        return self._value

    def update(self, bar: Bar) -> None:
        c, oi = bar.close, bar.open_interest
        if not (oi > 0.0 and math.isfinite(oi)):
            self._value *= 1.0 - self._alpha
            self._prev_close = c
            return

        if self._prev_close is None or self._prev_oi is None:
            self._prev_close = c
            self._prev_oi = oi
            self._ema_oi = oi
            return

        self._count += 1
        self._ema_oi = (1.0 - self._alpha_base) * self._ema_oi + self._alpha_base * oi
        rel_doi = (
            max(-0.10, min(0.10, (oi - self._prev_oi) / max(self._ema_oi, 1.0))) / 0.10
        )
        price_sign = (
            1.0 if c > self._prev_close else (-1.0 if c < self._prev_close else 0.0)
        )

        self._value = (1.0 - self._alpha) * self._value + self._alpha * (
            price_sign * rel_doi
        )
        self._prev_close = c
        self._prev_oi = oi
