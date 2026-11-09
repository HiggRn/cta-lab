from pathlib import Path

import numpy as np
import pandas as pd

from baselines import EwmaMonthlyRiskModel, MonthlyTsmomAlpha
from common import (
    AlphaModel,
    Bar,
    DailySchedule,
    EqualRiskPCM,
    FirstAllTradableDayMonthlySchedule,
    HistoryContext,
    InverseVolPCM,
    ModularStrategy,
    PortfolioConstructionModel,
    RiskModel,
    SampleCovarianceRiskModel,
    Strategy,
    VolTargetPCM,
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


# Daily TSMOM at 11% vol: the strongest result after costs
#
# Fixed before testing, not tuned: vol floor 3%/yr, trade buffer 20% of the target.
MIN_VOL = 0.03
BUFFER = 0.20


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


class TradeBufferPCM(PortfolioConstructionModel):
    """Wraps a PCM with a no-trade band around each target.

    The band is +-`buffer` x |target|. A position already inside the band is left alone;
    one outside is traded only to the nearest edge. Sign flips and exits always trade.
    """

    def __init__(self, base: PortfolioConstructionModel, buffer: float = BUFFER):
        self.base = base
        self.buffer = buffer
        self._held: dict[str, float] = {}

    def create_targets(self, signals, cov):
        targets = self.base.create_targets(signals, cov)
        final = {}
        for s, target in targets.items():
            band = self.buffer * abs(target)
            final[s] = float(np.clip(self._held.get(s, 0.0), target - band, target + band))
        self._held = final
        return final


class DailyTsmomVolTargetStrategy(ModularStrategy):
    """The daily 252d TSMOM example, scaled to 11% ex-ante vol, with two cost controls.

    1. Vol target: the report's main finding is that portfolio-level risk control helps.
       The example runs at about 1x gross and 4% vol; scaling to 11% makes it comparable.
    2. Vol floor: stops leverage piling into near-zero-vol contracts (financing cost).
    3. Trade buffer: daily rebalancing chases small changes in vol estimates; the band
       roughly halves turnover.
    """

    def __init__(self):
        super().__init__(
            alpha_model=DailyTsmomAlpha(lookback=252),
            risk_model=VolFloorRiskModel(
                SampleCovarianceRiskModel(periods_per_year=252, min_periods=20)
            ),
            pcm=TradeBufferPCM(VolTargetPCM(InverseVolPCM(1.0), target_vol=0.11)),
            schedule=DailySchedule(),
            risk_lookback=60,
        )


class FloorEwmaVolTargetStrategy(ModularStrategy):
    """'Vol target 11%, EWMA risk' baseline (the report's best risk model) with the 3% vol floor.

    The EWMA covariance forecasts risk best (report Section 7.1), but under 1/vol sizing it
    puts a lot of notional into short-rate futures and 2y notes. The floor caps that.
    """

    def __init__(self):
        super().__init__(
            alpha_model=MonthlyTsmomAlpha((12,)),
            risk_model=VolFloorRiskModel(EwmaMonthlyRiskModel()),
            pcm=VolTargetPCM(EqualRiskPCM(0.40), target_vol=0.11),
            schedule=FirstAllTradableDayMonthlySchedule(),
            risk_lookback=None,
        )


class SleeveMixStrategy(Strategy):
    """Runs several strategies side by side and holds a fixed-weight mix of their targets.

    Every sleeve sees every bar; the mix rebalances whenever any sleeve does.
    """

    def __init__(self, sleeves: list[Strategy], weights: list[float]):
        self.sleeves = sleeves
        self.weights = weights
        self._targets: list[dict[str, float]] = [{} for _ in sleeves]

    def on_bar(self, bars, context):
        changed = False
        for i, sleeve in enumerate(self.sleeves):
            out = sleeve.on_bar(bars, context)
            if out is not None:
                self._targets[i] = out
                changed = True
        if not changed:
            return None
        mix: dict[str, float] = {}
        for w, targets in zip(self.weights, self._targets):
            for s, x in targets.items():
                mix[s] = mix.get(s, 0.0) + w * x
        return mix


class MixedSpeedStrategy(SleeveMixStrategy):
    """50/50 mix of the monthly EWMA vol target (with floor) and the daily 11%-vol TSMOM.

    The two trend speeds and rebalance frequencies are not perfectly correlated, so the mix
    smooths drawdowns. CTAs blend horizons for the same reason (report Section 6).
    """

    def __init__(self):
        super().__init__(
            [FloorEwmaVolTargetStrategy(), DailyTsmomVolTargetStrategy()], [0.5, 0.5]
        )


# Asset-class risk parity (midterm report, Section 10, idea 3)

ASSET_MAP_PATH = Path(__file__).resolve().parent / "data" / "processed" / "AssetMapCsv.csv"


def _asset_classes() -> dict[str, str]:
    return pd.read_csv(ASSET_MAP_PATH, index_col="ID")["AssetClass"].to_dict()


def _dense_cov(cov: pd.DataFrame, names: list[str]) -> np.ndarray:
    """Covariance for `names`, with unknown pairs treated as uncorrelated."""
    c = cov.reindex(index=names, columns=names).to_numpy(dtype=float)
    diag = np.diag(c).copy()
    c = np.where(np.isnan(c), 0.0, c)
    np.fill_diagonal(c, np.nan_to_num(diag))
    return c


class AssetClassBudgetPCM(PortfolioConstructionModel):
    """Wraps a PCM so each asset class (COMM, EQ, FI, FX) carries the same stand-alone risk.

    Each class's sleeve is scaled to the average sleeve volatility, so the book's overall
    size stays about the same; an outer VolTargetPCM sets the final level.
    """

    def __init__(self, base: PortfolioConstructionModel, classes: dict[str, str] | None = None):
        self.base = base
        self.classes = classes if classes is not None else _asset_classes()

    def create_targets(self, signals, cov):
        targets = self.base.create_targets(signals, cov)
        if not targets:
            return targets
        groups: dict[str, list[str]] = {}
        for s in targets:
            groups.setdefault(self.classes.get(s, "OTHER"), []).append(s)
        sleeve_vol = {}
        for cls, names in groups.items():
            w = np.array([targets[s] for s in names])
            sleeve_vol[cls] = float(np.sqrt(max(w @ _dense_cov(cov, names) @ w, 0.0)))
        live = [v for v in sleeve_vol.values() if v > 0]
        if not live:
            return targets
        level = float(np.mean(live))
        out = {}
        for cls, names in groups.items():
            scale = level / sleeve_vol[cls] if sleeve_vol[cls] > 0 else 0.0
            out.update({s: targets[s] * scale for s in names})
        return out


class AssetClassRiskParityStrategy(ModularStrategy):
    """12m TSMOM, 40% / vol per contract, equal risk per asset class, scaled to 11%
    ex-ante vol with the EWMA monthly covariance (12-month half-life).

    Commodities are 26 of the 58 contracts, so under 1/N they supply about half the
    risk. Equal class budgets spread risk across the four classes.
    """

    def __init__(self):
        super().__init__(
            alpha_model=MonthlyTsmomAlpha((12,)),
            risk_model=EwmaMonthlyRiskModel(),
            pcm=VolTargetPCM(AssetClassBudgetPCM(EqualRiskPCM(0.40)), target_vol=0.11),
            schedule=FirstAllTradableDayMonthlySchedule(),
            risk_lookback=None,
        )


# Registry of user strategies to be evaluated in main.py
USER_STRATEGIES: dict[str, type[Strategy]] = {
    "Daily 252d TSMOM, inverse vol": DailyTsmomStrategy,
    "Asset-class risk parity (EWMA)": AssetClassRiskParityStrategy,
    "Daily TSMOM 11% vol (floor, buffer)": DailyTsmomVolTargetStrategy,
    "EWMA vol target + floor": FloorEwmaVolTargetStrategy,
    "Mix 50/50: EWMA + floor and daily 11% vol": MixedSpeedStrategy,
}
