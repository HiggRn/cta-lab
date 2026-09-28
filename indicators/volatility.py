import math

import pandas as pd

from common import Bar, Indicator


class Volatility(Indicator):
    """
    Moskowitz et al. (2012) EWMA ex-ante annualized volatility:
    Center of mass = 60 days (delta = 60/61), annualized by factor 261.
    """

    def __init__(
        self,
        com: float = 60.0,
        annual_factor: float = 261.0,
        min_periods: int = 60,
    ):
        self.decay = com / (com + 1.0)
        self.alpha = 1.0 - self.decay
        self.annual_factor = annual_factor
        self.min_periods = min_periods
        self._last_close: float | None = None
        self._mean: float = 0.0
        self._var: float = 0.0
        self._count: int = 0

    def on_bar(self, bar: Bar) -> None:
        price = bar.close
        if pd.isna(price) or price <= 0:
            return

        if self._last_close is not None and self._last_close > 0:
            ret = (price / self._last_close) - 1.0
            self._count += 1
            if self._count == 1:
                self._mean = ret
                self._var = 0.0
            else:
                self._mean = self.decay * self._mean + self.alpha * ret
                self._var = self.decay * self._var + self.alpha * (
                    (ret - self._mean) ** 2
                )

        self._last_close = price

    @property
    def value(self) -> float | None:
        if self._count < self.min_periods or self._var <= 1e-12:
            return None
        return math.sqrt(self.annual_factor * self._var)
