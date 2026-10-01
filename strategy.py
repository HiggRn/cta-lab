import numpy as np
import pandas as pd

from common import (
    AlphaModel,
    Bar,
    HistoryContext,
    PortfolioConstructionModel,
    RiskModel,
    Strategy,
)


class ModularStrategy(Strategy):
    def __init__(
        self,
        alpha_model: AlphaModel,
        risk_model: RiskModel,
        pcm: PortfolioConstructionModel,
        risk_lookback: int = 60,
        rebalance: str = "monthly",
    ):
        if rebalance not in ("daily", "monthly"):
            raise ValueError(f"Unsupported rebalance: {rebalance}")
        self.alpha_model = alpha_model
        self.risk_model = risk_model
        self.pcm = pcm
        self.risk_lookback = risk_lookback
        self.rebalance = rebalance

    def on_bar(
        self, bars: dict[str, Bar], context: HistoryContext
    ) -> dict[str, float] | None:
        if not bars:
            return None

        # alpha runs on every bar so it can keep its own state
        signals = self.alpha_model.update(bars, context)

        if not self._is_rebalance_bar(bars):
            return None

        cov = self.risk_model.covariance(context.get_returns(self.risk_lookback))
        return self.pcm.create_targets(signals, cov)

    def _is_rebalance_bar(self, bars: dict[str, Bar]) -> bool:
        if self.rebalance == "daily":
            return True
        return any(b.is_calendar_month_end for b in bars.values())


class SampleCovarianceRiskModel(RiskModel):
    def __init__(self, periods_per_year: int = 252):
        self.periods_per_year = periods_per_year

    def covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        return returns.cov() * self.periods_per_year


class InverseVolPCM(PortfolioConstructionModel):
    def __init__(self, target_gross_leverage: float = 1.0):
        self.target_gross_leverage = target_gross_leverage

    def create_targets(
        self, signals: dict[str, float], cov: pd.DataFrame
    ) -> dict[str, float]:
        vol = pd.Series(np.sqrt(np.diag(cov)), index=cov.index)

        raw = {}
        for symbol, signal in signals.items():
            if signal == 0.0 or symbol not in vol.index:
                continue
            if pd.notna(vol[symbol]) and vol[symbol] > 1e-4:
                raw[symbol] = signal / vol[symbol]

        gross = sum(abs(w) for w in raw.values())
        if gross == 0.0:
            return {}

        scale = self.target_gross_leverage / gross
        return {s: w * scale for s, w in raw.items()}
