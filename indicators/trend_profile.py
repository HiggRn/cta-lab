import math
from collections import deque

from common import Bar, Indicator

from .ewma_vol import EwmaVolatility

Window = tuple[int, int]


class TrendProfile(Indicator):
    """Scores the price history one piece at a time.

    Each window (a, b) is the move from b bars ago to a bars ago, divided by what a
    random walk with today's volatility would typically move in b - a bars, and capped
    at +-`cap`. `follow` windows count for the trend, `fade` windows against it.
    The value is the average over all windows.
    """

    def __init__(
        self,
        follow: tuple[Window, ...],
        fade: tuple[Window, ...] = (),
        vol_span: int = 60,
        cap: float = 2.0,
    ):
        self.windows = list(follow) + list(fade)
        self.signs = [1.0] * len(follow) + [-1.0] * len(fade)
        self.cap = cap
        self._vol = EwmaVolatility(span=vol_span)
        self._logs: deque[float] = deque(maxlen=max(b for _, b in self.windows) + 1)

    @property
    def is_ready(self) -> bool:
        return len(self._logs) == self._logs.maxlen and self._vol.is_ready

    @property
    def vol(self) -> float:
        """Daily volatility the scores are measured in."""
        return self._vol.value

    def scores(self) -> list[float]:
        """Capped z-score of each window, follow windows first."""
        logs, vol = self._logs, self._vol.value
        out = []
        for a, b in self.windows:
            z = (logs[-1 - a] - logs[-1 - b]) / (vol * math.sqrt(b - a))
            out.append(max(-self.cap, min(self.cap, z)))
        return out

    @property
    def value(self) -> float:
        return sum(s * z for s, z in zip(self.signs, self.scores())) / len(self.windows)

    def update(self, bar: Bar) -> None:
        if math.isnan(bar.close) or bar.close <= 0.0:
            return
        self._vol.update(bar)
        self._logs.append(math.log(bar.close))
