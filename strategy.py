import numpy as np

from common import (
    AlphaModel,
    Bar,
    DailySchedule,
    HistoryContext,
    InverseVolPCM,
    ModularStrategy,
    SampleCovarianceRiskModel,
    Strategy,
)
from indicators.momentum import Momentum

# Example Strategy: Daily 252d TSMOM


class DailyTsmomAlpha(AlphaModel):
    """Sign of 252-bar rolling return for each symbol."""

    def __init__(self, lookback: int = 252):
        self.lookback = lookback
        self._momentum: dict[str, Momentum] = {}

    def update(self, bars: dict[str, Bar], context: HistoryContext) -> dict[str, float]:
        signals = {}
        for symbol, bar in bars.items():
            if symbol not in self._momentum:
                self._momentum[symbol] = Momentum(self.lookback)
            ind = self._momentum[symbol]
            ind.update(bar)
            if ind.is_ready:
                signals[symbol] = float(np.sign(ind.value))
        return signals


class DailyTsmomStrategy(ModularStrategy):
    """Daily rebalanced 252d TSMOM with 60-day sample covariance and inverse-vol sizing."""

    def __init__(self, lookback: int = 252, risk_lookback: int = 60):
        super().__init__(
            alpha_model=DailyTsmomAlpha(lookback=lookback),
            risk_model=SampleCovarianceRiskModel(periods_per_year=252, min_periods=20),
            pcm=InverseVolPCM(target_gross_leverage=1.0),
            schedule=DailySchedule(),
            risk_lookback=risk_lookback,
        )


# Registry of user strategies to be evaluated in main.py
USER_STRATEGIES: dict[str, type[Strategy]] = {
    "Daily 252d TSMOM, inverse vol": DailyTsmomStrategy,
}
