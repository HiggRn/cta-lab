from collections import deque

import numpy as np
import pandas as pd

from common import (
    AlphaModel,
    Bar,
    HistoryContext,
    InverseVolPCM,
    ModularStrategy,
    PortfolioConstructionModel,
    RebalanceSchedule,
    RiskModel,
    Strategy,
    VolTargetPCM,
)
from config import ASSET_MAP_PATH
from indicators import TrendTStat

# Fixed before testing, not tuned: vol floor 3%/yr.
MIN_VOL = 0.03


class VolFloorRiskModel(RiskModel):
    """Wraps a risk model and raises every volatility below `min_vol` to `min_vol`.

    Correlations are kept. Stops very quiet contracts (short-rate futures, 2y notes)
    from getting huge positions under 1/vol sizing, which mostly cost financing.
    """

    def __init__(self, base: RiskModel, min_vol: float = MIN_VOL):
        self.base = base
        self.min_vol = min_vol

    def covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        cov = self.base.covariance(returns)
        vol = np.sqrt(np.diag(cov))
        with np.errstate(invalid="ignore", divide="ignore"):
            scale = np.where(vol > 0, np.maximum(vol, self.min_vol) / vol, np.nan)
        return cov * np.outer(scale, scale)


# Asset-class risk parity (midterm report, Section 10, idea 3)


def _asset_classes() -> dict[str, str]:
    return pd.read_csv(ASSET_MAP_PATH, index_col="ID", keep_default_na=False)[
        "AssetClass"
    ].to_dict() | {"EC": "FI"}


# KV1 / KV2: the baseline with a few changes (README, "Final strategies: KV1 and KV2")
#
# The baseline (RollingVolTsmomStrategy in baselines.py): sign of the 12-month return,
# 1/vol sizing, scaled to 11% ex-ante vol, rebuilt every 21 trading days. KV1 changes four
# things and KV2 two more. The vol floor is the 3% one from the daily 11% vol strategy;
# the rest are below.
NO_SHORT = ("EQ",)  # equity indices are never shorted
CYCLE = 21  # trading days each tranche is held
TRANCHE_DAYS = (0, 5, 10, 15)  # days of the cycle on which one tranche is rebuilt


class NoShortAlpha(AlphaModel):
    """Wraps an alpha and sets short signals to zero in the asset classes in `flat`."""

    def __init__(
        self,
        base: AlphaModel,
        flat: tuple[str, ...] = NO_SHORT,
        classes: dict[str, str] | None = None,
    ):
        self.base = base
        self.flat = flat
        self.classes = classes if classes is not None else _asset_classes()

    def update(self, bars: dict[str, Bar], context: HistoryContext) -> dict[str, float]:
        signals = self.base.update(bars, context)
        return {
            s: 0.0 if x < 0.0 and self.classes.get(s) in self.flat else x
            for s, x in signals.items()
        }


class TrendStrengthAlpha(AlphaModel):
    """t-stat of the 252-bar return, capped at +-`cap` and divided by it.

    A clear trend gets +-1, same as the sign. A 12-month move that is small next to the
    contract's own noise gets a smaller position instead of a full one.
    """

    def __init__(self, lookback: int = 252, cap: float = 1.0):
        self.lookback = lookback
        self.cap = cap
        self._tstat: dict[str, TrendTStat] = {}

    def update(self, bars: dict[str, Bar], context: HistoryContext) -> dict[str, float]:
        signals = {}
        for symbol, bar in bars.items():
            if symbol not in self._tstat:
                self._tstat[symbol] = TrendTStat(self.lookback)
            ind = self._tstat[symbol]
            ind.update(bar)
            if ind.is_ready:
                signals[symbol] = float(np.clip(ind.value / self.cap, -1.0, 1.0))
        return signals


class EwmaVolRiskModel(RiskModel):
    """Sample correlations of the returns passed in, with EWMA vols (centre of mass 60
    days) on the diagonal.

    The 12-month vol is slow: after a quiet year it keeps positions big well into a
    selloff. Correlations stay on the full window, where they are better estimated.
    """

    def __init__(self, com: float = 60.0, periods_per_year: int = 252):
        self.com = com
        self.periods_per_year = periods_per_year

    def covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        cov = returns.cov(min_periods=20) * self.periods_per_year
        ewma = returns.ewm(com=self.com, min_periods=20).std().iloc[-1]
        with np.errstate(invalid="ignore", divide="ignore"):
            scale = (
                ewma.to_numpy() * np.sqrt(self.periods_per_year) / np.sqrt(np.diag(cov))
            )
        return cov * np.outer(scale, scale)


class TrancheSchedule(RebalanceSchedule):
    """Rebalance on trading days `days` of every `cycle`-day cycle."""

    def __init__(self, cycle: int = CYCLE, days: tuple[int, ...] = TRANCHE_DAYS):
        self.cycle = cycle
        self.days = days
        self._count = -1

    def should_rebalance(self, bars: dict[str, Bar], context: HistoryContext) -> bool:
        if not bars:
            return False
        self._count += 1
        return self._count % self.cycle in self.days


class TranchePCM(PortfolioConstructionModel):
    """Wraps a PCM and holds the average of its last `parts` targets.

    With `TrancheSchedule` these are four books with a quarter of the capital each,
    every one rebuilt every 21 trading days, about a week apart. A single monthly book
    depends a lot on which day of the month it happens to trade.
    """

    def __init__(
        self, base: PortfolioConstructionModel, parts: int = len(TRANCHE_DAYS)
    ):
        self.base = base
        self._recent: deque[dict[str, float]] = deque(maxlen=parts)

    def create_targets(self, signals, cov):
        self._recent.append(self.base.create_targets(signals, cov))
        book: dict[str, float] = {}
        for targets in self._recent:
            for s, w in targets.items():
                book[s] = book.get(s, 0.0) + w / self._recent.maxlen
        return book


class KVStrategy(ModularStrategy):
    """12-month sign, no equity shorts, EWMA vol with the 3% floor, four weekly
    tranches, 11% vol.
    """

    def __init__(
        self,
        alpha_model: AlphaModel | None = None,
        pcm: PortfolioConstructionModel | None = None,
    ):
        super().__init__(
            alpha_model=NoShortAlpha(TrendStrengthAlpha()),
            risk_model=VolFloorRiskModel(EwmaVolRiskModel()),
            pcm=TranchePCM(VolTargetPCM(InverseVolPCM(), target_vol=0.11)),
            schedule=TrancheSchedule(),
            risk_lookback=252,
        )


# Registry of user strategies to be evaluated in main.py
USER_STRATEGIES: dict[str, type[Strategy]] = {
    "KV": KVStrategy,
}
