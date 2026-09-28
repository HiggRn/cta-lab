from __future__ import annotations

import math
from collections import defaultdict

from common import Bar, Strategy
from indicators import MonthlyReturn
from indicators.volatility import Volatility


class TimeSeriesMomentumStrategy(Strategy):
    def __init__(
        self,
        target_vol: float = 0.40,
        lookback_months: int = 12,
        normalize_by_count: bool = True,
    ):
        """
        :param target_vol: Target annualized volatility per asset (e.g., 0.40 for 40%).
        :param lookback_months: Lookback horizon in months for momentum evaluation (default: 12).
        :param normalize_by_count: Whether to scale positions by the number of available assets S_t.
               True corresponds to the equal-weighted multi-asset portfolio in Moskowitz et al. (2012)
               following Eq. (5) (i.e., (1 / S_t) * sum(pos)).
               False directly outputs standalone nominal leverage per asset.
        """
        self.target_vol = target_vol
        self.lookback_months = lookback_months
        self.normalize_by_count = normalize_by_count

        self.monthly_returns: dict[str, MonthlyReturn] = defaultdict(
            lambda: MonthlyReturn(lookback=self.lookback_months)
        )
        self.vols: dict[str, Volatility] = defaultdict(Volatility)

    def on_bar(self, bars: dict[str, Bar]) -> dict[str, float] | None:
        if not bars:
            return None

        # 1. feed bar into indicators
        for s, bar in bars.items():
            self.monthly_returns[s].on_bar(bar)
            self.vols[s].on_bar(bar)

        # 2. trigger only at month end
        if not any(b.is_calendar_month_end for b in bars.values()):
            return None

        # 3. position generation
        positions: dict[str, float] = {}
        for s in bars:
            mr = self.monthly_returns[s]
            vol = self.vols[s].value

            # must collect 12 monthly returns
            if len(mr.history) < self.lookback_months or vol is None or vol <= 1e-4:
                continue

            # sign of cum returns for past 12 months
            cum_ret = math.prod(1.0 + r for r in mr.history) - 1.0
            signal = 1.0 if cum_ret > 0 else (-1.0 if cum_ret < 0 else 0.0)
            if signal == 0.0:
                continue

            positions[s] = signal * self.target_vol / vol

        if not positions:
            return {}

        # 4. portfolio construction
        scale = (1.0 / len(positions)) if self.normalize_by_count else 1.0
        return {s: pos * scale for s, pos in positions.items()}
