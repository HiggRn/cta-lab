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
    ):
        self.alpha_model = alpha_model
        self.risk_model = risk_model
        self.pcm = pcm
        self.risk_lookback = risk_lookback

    def on_bar(
        self, bars: dict[str, Bar], context: HistoryContext
    ) -> dict[str, float] | None:
        if not bars:
            return None

        signals = self.alpha_model.update(bars, context)

        returns_window = context.get_returns(self.risk_lookback)
        risk_metrics = self.risk_model.calculate_risk(returns_window)

        return self.pcm.create_targets(bars, signals, risk_metrics)


class SampleCovarianceRiskModel(RiskModel):
    def __init__(self, annualize_factor: float = 252**0.5):
        self.annualize_factor = annualize_factor

    def calculate_risk(
        self, returns_history: pd.DataFrame
    ) -> tuple[pd.Series, pd.DataFrame]:
        vol = returns_history.std(ddof=1) * self.annualize_factor
        cov = returns_history.cov() * (self.annualize_factor**2)
        return vol, cov


class RiskBudgetingPCM(PortfolioConstructionModel):
    def __init__(
        self,
        frequency: str = "monthly",
        target_gross_leverage: float = 1.0,
    ):
        if frequency not in ("daily", "monthly"):
            raise ValueError(f"Unsupported frequency: {frequency}")
        self.frequency = frequency
        self.target_gross_leverage = target_gross_leverage

    def create_targets(
        self,
        bars: dict[str, Bar],
        signals: dict[str, float],
        risk_metrics: tuple[pd.Series, pd.DataFrame],
    ) -> dict[str, float] | None:
        if self.frequency == "monthly":
            is_month_end = any(b.is_calendar_month_end for b in bars.values())
            if not is_month_end:
                return None

        vol, _ = risk_metrics
        raw_weights = {}

        for symbol, signal in signals.items():
            if signal == 0.0 or symbol not in vol:
                continue
            asset_vol = vol[symbol]
            if pd.notna(asset_vol) and asset_vol > 1e-4:
                raw_weights[symbol] = signal / asset_vol

        if not raw_weights:
            return {s: 0.0 for s in bars.keys()}

        total_risk = sum(abs(w) for w in raw_weights.values())
        if total_risk == 0.0:
            return {s: 0.0 for s in bars.keys()}

        scale = self.target_gross_leverage / total_risk
        return {s: w * scale for s, w in raw_weights.items()}
