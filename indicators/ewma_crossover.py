import math

from common import Bar, Indicator
from indicators import ATR


class EwmaCrossover(Indicator):
    """Single-scale ATR-normalized EWMA crossover z-score: (EMA_fast - EMA_slow) / (ATR * norm)."""

    def __init__(self, fast_span: int, slow_span: int, atr_span: int = 32):
        self._af = 2.0 / (fast_span + 1.0)
        self._as = 2.0 / (slow_span + 1.0)
        bf, bs = 1.0 - self._af, 1.0 - self._as
        var_scale = (
            1.0 / (1.0 - bf * bf) + 1.0 / (1.0 - bs * bs) - 2.0 / (1.0 - bf * bs)
        )
        self._norm = math.sqrt(max(var_scale, 1e-8))

        self._min_bars = slow_span // 2
        self._count = 0
        self._fast = 0.0
        self._slow = 0.0
        self._atr = ATR(atr_span)

    @property
    def is_ready(self) -> bool:
        return self._count >= self._min_bars and self._atr.is_ready

    @property
    def value(self) -> float:
        return (self._fast - self._slow) / (self._atr.value * self._norm)

    def update(self, bar: Bar) -> None:
        c = bar.close
        self._atr.update(bar)
        self._count += 1
        if self._count == 1:
            self._fast = c
            self._slow = c
        else:
            self._fast = (1.0 - self._af) * self._fast + self._af * c
            self._slow = (1.0 - self._as) * self._slow + self._as * c
