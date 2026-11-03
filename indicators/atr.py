import math

from common import Bar, Indicator


class ATR(Indicator):
    """Exponential Moving Average True Range (ATR) in price units."""

    def __init__(self, span: int = 126):
        if span <= 1:
            raise ValueError(f"span {span} must be > 1")
        self.span = span
        self.alpha = 2.0 / (span + 1.0)
        self._prev_close: float | None = None
        self._atr: float | None = None
        self._count: int = 0

    @property
    def is_ready(self) -> bool:
        return self._count >= self.span and self._atr is not None and self._atr > 0.0

    @property
    def value(self) -> float:
        return float(self._atr) if self._atr is not None else 0.0

    def update(self, bar: Bar) -> None:
        c = bar.close
        if math.isnan(c) or c <= 0.0:
            return

        h = bar.high if (not math.isnan(bar.high) and bar.high > 0.0) else c
        l = bar.low if (not math.isnan(bar.low) and bar.low > 0.0) else c
        h = max(h, l, c)
        l = min(l, c)

        if self._prev_close is None:
            tr = h - l
        else:
            tr = max(h - l, abs(h - self._prev_close), abs(l - self._prev_close))
        self._prev_close = c

        if self._atr is None:
            self._atr = tr
        else:
            self._atr = (1.0 - self.alpha) * self._atr + self.alpha * tr

        self._count += 1
