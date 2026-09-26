from collections import defaultdict

import pandas as pd

from common import Bar, Strategy
from indicators import MonthlyReturn


class MonthlyReturnValidationStrategy(Strategy):
    def __init__(self):
        self.indicators: dict[str, MonthlyReturn] = defaultdict(MonthlyReturn)
        self.recorded_monthly_returns: dict[pd.Timestamp, dict[str, float]] = {}

    def on_bar(self, bars: dict[str, Bar]) -> dict[str, float] | None:
        if not bars:
            return None

        today_rets = {}
        current_date = None
        for symbol, bar in bars.items():
            self.indicators[symbol].on_bar(bar)
            current_date = current_date or bar.date
            ret = self.indicators[symbol].value
            if ret is not None:
                today_rets[symbol] = ret

        if today_rets:
            self.recorded_monthly_returns[current_date] = today_rets
        return None
