import math

from common import Bar, Indicator


class YangZhangVol(Indicator):
    """Single-halflife EWMA Yang-Zhang (2000) annualized OHLC volatility."""

    def __init__(
        self, halflife: float = 20.0, min_bars: int = 20, periods_per_year: int = 252
    ):
        self._alpha = 1.0 - math.exp(-math.log(2.0) / halflife)
        self._min_bars = min_bars
        self._periods = periods_per_year

        n_eff = max(2.0 * halflife, 10.0)
        self._k = 0.34 / (1.34 + (n_eff + 1.0) / (n_eff - 1.0))

        self._prev_close: float | None = None
        self._count = 0
        self._mean_o = 0.0
        self._mean_c = 0.0
        self._var_o = 0.0
        self._var_c = 0.0
        self._var_rs = 0.0

    @property
    def is_ready(self) -> bool:
        return self._count >= self._min_bars

    @property
    def value(self) -> float:
        daily_var = self._var_o + self._k * self._var_c + (1.0 - self._k) * self._var_rs
        return math.sqrt(max(daily_var, 0.0) * self._periods)

    def update(self, bar: Bar) -> None:
        c = bar.close
        if self._prev_close is None:
            self._prev_close = c
            return

        o = bar.open if bar.open > 0.0 else c
        h = max(bar.high if bar.high > 0.0 else c, o, c)
        l = min(bar.low if bar.low > 0.0 else c, o, c)

        ret_o = max(-0.25, min(0.25, math.log(o / self._prev_close)))
        ret_c = max(-0.25, min(0.25, math.log(c / o)))
        u, d = math.log(h / o), math.log(l / o)
        rs = max(0.0, u * (u - ret_c) + d * (d - ret_c))

        self._prev_close = c
        self._count += 1
        a = self._alpha

        if self._count == 1:
            self._mean_o, self._mean_c = ret_o, ret_c
            self._var_o, self._var_c, self._var_rs = ret_o * ret_o, ret_c * ret_c, rs
            return

        do, dc = ret_o - self._mean_o, ret_c - self._mean_c
        self._mean_o += a * do
        self._mean_c += a * dc

        self._var_o = (1.0 - a) * self._var_o + a * do * (ret_o - self._mean_o)
        self._var_c = (1.0 - a) * self._var_c + a * dc * (ret_c - self._mean_c)
        self._var_rs = (1.0 - a) * self._var_rs + a * rs
