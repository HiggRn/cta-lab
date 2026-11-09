import numpy as np
import pandas as pd

from common import (
    AlphaModel,
    Bar,
    DailySchedule,
    HistoryContext,
    InverseVolPCM,
    ModularStrategy,
    PortfolioConstructionModel,
    RiskModel,
    SampleCovarianceRiskModel,
    Strategy,
)
from indicators import Momentum, YangZhangVolatility


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


class YangZhangRiskModel(RiskModel):
    """Sample correlation matrix scaled by Yang-Zhang OHLC volatilities on the diagonal."""

    def __init__(
        self,
        lookback: int = 60,
        periods_per_year: int = 252,
        min_periods: int = 20,
    ):
        self.lookback = lookback
        self.periods_per_year = periods_per_year
        self.min_periods = min_periods
        self._yz: dict[str, YangZhangVolatility] = {}

    def update(self, bars: dict[str, Bar]) -> None:
        for symbol, bar in bars.items():
            if symbol not in self._yz:
                self._yz[symbol] = YangZhangVolatility(
                    lookback=self.lookback,
                    min_periods=self.min_periods,
                    periods_per_year=self.periods_per_year,
                )
            self._yz[symbol].update(bar)

    def covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        cov = returns.cov(min_periods=self.min_periods) * self.periods_per_year
        cc_vol = np.sqrt(np.maximum(np.diag(cov), 0.0))

        scale = np.ones(len(cov.columns), dtype=float)
        for i, sym in enumerate(cov.columns):
            ind = self._yz.get(sym)
            if ind is not None and ind.is_ready and cc_vol[i] > 1e-8:
                scale[i] = ind.value / cc_vol[i]

        scaled_cov = cov.to_numpy(dtype=float) * np.outer(scale, scale)
        return pd.DataFrame(scaled_cov, index=cov.index, columns=cov.columns)


class CorrelationAdjustedPCM(PortfolioConstructionModel):
    """Continuous correlation-crowding adjusted inverse-vol sizing: w_i ~ s_i / (vol_i * sum_j max(corr_ij, 0))."""

    def __init__(self, target_gross_leverage: float = 1.0):
        self.target_gross_leverage = target_gross_leverage

    def create_targets(
        self, signals: dict[str, float], cov: pd.DataFrame
    ) -> dict[str, float]:
        vol = pd.Series(np.sqrt(np.maximum(np.diag(cov), 0.0)), index=cov.index)
        live = [
            s
            for s, sig in signals.items()
            if sig != 0.0 and s in vol.index and pd.notna(vol[s]) and vol[s] > 1e-4
        ]
        if not live:
            return {}

        sub_cov = np.nan_to_num(cov.loc[live, live].to_numpy(dtype=float), nan=0.0)
        std = np.sqrt(np.diag(sub_cov))
        s_vec = np.array([signals[s] for s in live], dtype=float)

        corr = np.clip(sub_cov / np.outer(std, std), -1.0, 1.0)
        np.fill_diagonal(corr, 1.0)

        crowding = np.sum(np.maximum(corr, 0.0), axis=1)
        raw = s_vec / (std * crowding)

        scale = self.target_gross_leverage / float(np.sum(np.abs(raw)))
        return {live[i]: float(raw[i] * scale) for i in range(len(live))}


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


class DailyTsmomCorrAdjustedStrategy(ModularStrategy):
    """Daily rebalanced 252d TSMOM with Yang-Zhang OHLC risk and correlation-adjusted sizing."""

    def __init__(self, lookback: int = 252, risk_lookback: int = 60):
        self._yz_risk = YangZhangRiskModel(
            lookback=risk_lookback, periods_per_year=252, min_periods=20
        )
        super().__init__(
            alpha_model=DailyTsmomAlpha(lookback=lookback),
            risk_model=self._yz_risk,
            pcm=CorrelationAdjustedPCM(target_gross_leverage=1.0),
            schedule=DailySchedule(),
            risk_lookback=risk_lookback,
        )

    def on_bar(
        self, bars: dict[str, Bar], context: HistoryContext
    ) -> dict[str, float] | None:
        self._yz_risk.update(bars)
        return super().on_bar(bars, context)


USER_STRATEGIES: dict[str, type[Strategy]] = {
    "Daily 252d TSMOM, inverse vol": DailyTsmomStrategy,
    "Daily 252d TSMOM, YZ + corr adjusted": DailyTsmomCorrAdjustedStrategy,
}
