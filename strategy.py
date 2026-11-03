import math

import numpy as np
import pandas as pd

from common import (
    AlphaModel,
    Bar,
    DailySchedule,
    EqualRiskPCM,
    HistoryContext,
    ModularStrategy,
    PortfolioConstructionModel,
    RiskModel,
    Strategy,
    VolTargetPCM,
)
from indicators.atr import ATR
from indicators.oi_anchor import OIVolumeAnchor

# Daily OI-Anchored ATR Breakout CTA Strategy


class OIAtrBreakoutAlpha(AlphaModel):
    """Combines OIVolumeAnchor and ATR into a volatility-normalized breakout z-score."""

    def __init__(self, span: int = 126, z_scale: float = 1.5):
        self.span = span
        self.z_scale = z_scale
        self._sqrt_span = math.sqrt(span)
        self._anchor: dict[str, OIVolumeAnchor] = {}
        self._atr: dict[str, ATR] = {}

    def update(self, bars: dict[str, Bar], context: HistoryContext) -> dict[str, float]:
        signals: dict[str, float] = {}
        for symbol, bar in bars.items():
            if symbol not in self._anchor:
                self._anchor[symbol] = OIVolumeAnchor(span=self.span)
                self._atr[symbol] = ATR(span=self.span)

            anchor_ind = self._anchor[symbol]
            atr_ind = self._atr[symbol]

            anchor_ind.update(bar)
            atr_ind.update(bar)

            if anchor_ind.is_ready and atr_ind.is_ready:
                denom = self._sqrt_span * max(atr_ind.value, 1e-8)
                z_score = (bar.close - anchor_ind.value) / denom
                signals[symbol] = math.tanh(z_score / self.z_scale)

        return signals


class DailyEwmaRiskModel(RiskModel):
    """Diagonal 60-day center-of-mass EWMA daily variance risk model."""

    def __init__(
        self, com: float = 60.0, min_days: int = 60, periods_per_year: int = 252
    ):
        self.com = com
        self.min_days = min_days
        self.periods_per_year = periods_per_year

    def covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        if len(returns) < self.min_days:
            return pd.DataFrame(np.nan, index=returns.columns, columns=returns.columns)
        var = (
            returns.ewm(com=self.com, min_periods=self.min_days).var().iloc[-1]
            * self.periods_per_year
        )
        return pd.DataFrame(
            np.diag(var), index=returns.columns, columns=returns.columns
        )


class TradeBufferedPCM(PortfolioConstructionModel):
    """Suppresses daily target weight updates smaller than `buffer_ratio` of full position size."""

    def __init__(
        self, base_pcm: PortfolioConstructionModel, buffer_ratio: float = 0.15
    ):
        self.base_pcm = base_pcm
        self.buffer_ratio = buffer_ratio
        self._last_targets: dict[str, float] = {}

    def create_targets(
        self, signals: dict[str, float], cov: pd.DataFrame
    ) -> dict[str, float]:
        active_signals = {s: sig for s, sig in signals.items() if abs(sig) > 1e-3}
        ideal = self.base_pcm.create_targets(active_signals, cov)
        if not ideal:
            self._last_targets = {}
            return {}

        final: dict[str, float] = {}
        for s, w_new in ideal.items():
            w_old = self._last_targets.get(s, 0.0)
            sig = abs(active_signals.get(s, 0.0))
            full_ref_weight = abs(w_new) / sig if sig > 1e-3 else abs(w_new)

            if abs(w_new - w_old) >= self.buffer_ratio * full_ref_weight:
                final[s] = w_new
            else:
                final[s] = w_old

        self._last_targets = final
        return final


class OIAtrBreakoutStrategy(ModularStrategy):
    """Daily OI-ATR Breakout CTA with 60d EWMA risk, 11% vol target, and 15% turnover buffer."""

    def __init__(
        self,
        span: int = 126,
        z_scale: float = 1.5,
        target_vol: float = 0.11,
        buffer_ratio: float = 0.15,
        risk_lookback: int = 252,
    ):
        base_pcm = VolTargetPCM(
            base_pcm=EqualRiskPCM(position_vol=0.40, min_assets=10),
            target_vol=target_vol,
        )
        super().__init__(
            alpha_model=OIAtrBreakoutAlpha(span=span, z_scale=z_scale),
            risk_model=DailyEwmaRiskModel(com=60.0, min_days=60, periods_per_year=252),
            pcm=TradeBufferedPCM(base_pcm=base_pcm, buffer_ratio=buffer_ratio),
            schedule=DailySchedule(),
            risk_lookback=risk_lookback,
        )


# Registry of user strategies to be evaluated in main.py
USER_STRATEGIES: dict[str, type[Strategy]] = {
    "Daily 126d OI-ATR Breakout": OIAtrBreakoutStrategy,
}
