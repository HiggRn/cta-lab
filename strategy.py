from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from baselines import EwmaMonthlyRiskModel, ExpandingMonthlyRiskModel, MonthlyTsmomAlpha
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


# Enhancement components: volatility floor, leverage caps, trade buffer, HRP
#
# Parameters below were fixed before testing (not tuned on any sample):
#   vol floor 3%/yr, at most 1x capital per contract, at most 5x gross
#   (about the in-sample baseline's peak), trade buffer 20% of the target position.
MIN_VOL = 0.03
MAX_WEIGHT = 1.0
MAX_GROSS = 5.0
BUFFER = 0.20


class VolFloorRiskModel(RiskModel):
    """Wraps a risk model and raises every volatility below `min_vol` to `min_vol`.

    Correlations are kept. Stops very quiet contracts (short-rate futures near zero
    rates) from getting huge positions under 1/vol sizing.
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


class LeverageCapPCM(PortfolioConstructionModel):
    """Wraps a PCM: clips each weight to +-`max_weight`, then scales the book down
    if gross leverage is above `max_gross`."""

    def __init__(
        self,
        base: PortfolioConstructionModel,
        max_weight: float = MAX_WEIGHT,
        max_gross: float = MAX_GROSS,
    ):
        self.base = base
        self.max_weight = max_weight
        self.max_gross = max_gross

    def create_targets(self, signals, cov):
        targets = {
            s: float(np.clip(w, -self.max_weight, self.max_weight))
            for s, w in self.base.create_targets(signals, cov).items()
        }
        gross = sum(abs(w) for w in targets.values())
        if gross > self.max_gross:
            targets = {s: w * self.max_gross / gross for s, w in targets.items()}
        return targets


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


class HrpPCM(PortfolioConstructionModel):
    """Hierarchical Risk Parity (Lopez de Prado, 2016) applied to the signed positions.

    1. Flip each contract's returns by its signal sign, so a short is treated as its own asset.
    2. Cluster contracts on correlation distance sqrt((1 - corr) / 2), single linkage.
    3. Split the cluster tree top-down, giving each half weight in inverse proportion
       to its variance. Weights sum to 1 and are then multiplied by the signal.
    Pairs without enough joint history count as uncorrelated.
    """

    def __init__(self, min_assets: int = 10):
        self.min_assets = min_assets

    def create_targets(self, signals, cov):
        variance = pd.Series(np.diag(cov), index=cov.index)
        live = [
            s
            for s, x in signals.items()
            if x != 0.0 and s in variance.index and variance[s] > 1e-8
        ]
        if len(live) < self.min_assets:
            return {}
        if len(live) == 1:
            return {live[0]: float(signals[live[0]])}

        sign = np.sign([signals[s] for s in live])
        c = cov.loc[live, live].to_numpy() * np.outer(sign, sign)
        vol = np.sqrt(np.diag(c))
        c = np.where(np.isnan(c), 0.0, c)
        np.fill_diagonal(c, vol**2)
        corr = np.clip(c / np.outer(vol, vol), -1.0, 1.0)
        dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
        np.fill_diagonal(dist, 0.0)
        order = list(leaves_list(linkage(squareform(dist, checks=False), "single")))

        weights = np.ones(len(live))
        clusters = [order]
        while clusters:
            next_level = []
            for cluster in clusters:
                if len(cluster) < 2:
                    continue
                left, right = cluster[: len(cluster) // 2], cluster[len(cluster) // 2 :]
                var_l, var_r = self._cluster_var(c, left), self._cluster_var(c, right)
                alpha = 1.0 - var_l / (var_l + var_r)
                weights[left] *= alpha
                weights[right] *= 1.0 - alpha
                next_level += [left, right]
            clusters = next_level

        return {s: float(w * signals[s]) for s, w in zip(live, weights)}

    @staticmethod
    def _cluster_var(c: np.ndarray, idx: list[int]) -> float:
        sub = c[np.ix_(idx, idx)]
        ivp = 1.0 / np.diag(sub)
        ivp /= ivp.sum()
        return float(ivp @ sub @ ivp)


# Enhanced strategies. Each starts from a course baseline and changes one thing,
# except the last, which combines them.


class _MonthlyStrategy(ModularStrategy):
    """Monthly rebalance on the course schedule, risk model sees all history."""

    def __init__(self, alpha, risk, pcm):
        super().__init__(
            alpha_model=alpha,
            risk_model=risk,
            pcm=pcm,
            schedule=FirstAllTradableDayMonthlySchedule(),
            risk_lookback=None,
        )


class FloorCapTsmomStrategy(_MonthlyStrategy):
    """Course baseline (12m TSMOM, 40% / vol, 1/N) with a 3% vol floor and leverage caps."""

    def __init__(self):
        super().__init__(
            MonthlyTsmomAlpha((12,)),
            VolFloorRiskModel(ExpandingMonthlyRiskModel()),
            LeverageCapPCM(EqualRiskPCM(0.40)),
        )


class BufferedTsmomStrategy(_MonthlyStrategy):
    """Course baseline with a 20% no-trade band to cut turnover."""

    def __init__(self):
        super().__init__(
            MonthlyTsmomAlpha((12,)),
            ExpandingMonthlyRiskModel(),
            TradeBufferPCM(EqualRiskPCM(0.40)),
        )


class FloorCapEwmaVolTargetStrategy(_MonthlyStrategy):
    """'Vol target 11%, EWMA risk' baseline with the vol floor and leverage caps."""

    def __init__(self):
        super().__init__(
            MonthlyTsmomAlpha((12,)),
            VolFloorRiskModel(EwmaMonthlyRiskModel()),
            LeverageCapPCM(VolTargetPCM(EqualRiskPCM(0.40), target_vol=0.11)),
        )


class HrpTsmomStrategy(_MonthlyStrategy):
    """12m TSMOM sized by HRP on the EWMA monthly covariance, scaled to 11% vol.

    Uses the vol floor and caps, since HRP's inverse-variance split would otherwise
    pile into the quietest contracts.
    """

    def __init__(self):
        super().__init__(
            MonthlyTsmomAlpha((12,)),
            VolFloorRiskModel(EwmaMonthlyRiskModel()),
            LeverageCapPCM(VolTargetPCM(HrpPCM(), target_vol=0.11)),
        )


class HrpBufferedTsmomStrategy(_MonthlyStrategy):
    """HRP TSMOM (vol floor, caps, 11% vol target) plus the 20% trade buffer."""

    def __init__(self):
        super().__init__(
            MonthlyTsmomAlpha((12,)),
            VolFloorRiskModel(EwmaMonthlyRiskModel()),
            TradeBufferPCM(LeverageCapPCM(VolTargetPCM(HrpPCM(), target_vol=0.11))),
        )


# Round 2: risk caps, asset-class budgets, continuous signal, strategy mix
#
# Fixed before testing: each contract at most 10% of portfolio variance, equal risk
# per asset class, signal = 12m t-stat capped at +-1, 50/50 mix of two 11%-vol sleeves.
MAX_RISK_SHARE = 0.10
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


class RiskCapPCM(PortfolioConstructionModel):
    """Wraps a PCM so no contract supplies more than `max_share` of portfolio variance.

    Risk share is the Euler contribution w_i (Cov w)_i / w'Cov w. Contracts over the cap
    are scaled down until none is (with fewer than 1.5 / max_share contracts the cap
    loosens to 1.5 / n). Total portfolio volatility is then restored, so the cap moves
    risk between contracts rather than cutting it.
    """

    def __init__(self, base: PortfolioConstructionModel, max_share: float = MAX_RISK_SHARE):
        self.base = base
        self.max_share = max_share

    def create_targets(self, signals, cov):
        targets = self.base.create_targets(signals, cov)
        if len(targets) < 2:
            return targets
        names = list(targets)
        w = np.array([targets[s] for s in names])
        c = _dense_cov(cov, names)
        start_var = w @ c @ w
        if not start_var > 0:
            return targets
        cap = max(self.max_share, 1.5 / len(names))
        for _ in range(100):
            var = w @ c @ w
            share = w * (c @ w) / var
            over = share > cap * (1 + 1e-6)
            if not over.any():
                break
            w[over] *= np.sqrt(cap / share[over])
        w *= np.sqrt(start_var / (w @ c @ w))
        return dict(zip(names, map(float, w)))


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


class ContinuousTsmomAlpha(MonthlyTsmomAlpha):
    """Trend strength instead of trend sign: t-stat of the last 12 monthly returns, capped at +-1.

    Weak trends get small positions, so the book stops flipping fully long/short on noise.
    """

    def direction(self, completed_returns: list[float]) -> float:
        r = np.asarray(completed_returns[-max(self.lookbacks):], dtype=float)
        sd = r.std(ddof=1)
        if not sd > 0:
            return float(np.sign(r.sum()))
        return float(np.clip(r.mean() / (sd / np.sqrt(len(r))), -1.0, 1.0))


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


class RiskCapEwmaStrategy(_MonthlyStrategy):
    """'EWMA vol target + floor & caps' with each contract capped at 10% of portfolio risk."""

    def __init__(self):
        super().__init__(
            MonthlyTsmomAlpha((12,)),
            VolFloorRiskModel(EwmaMonthlyRiskModel()),
            LeverageCapPCM(VolTargetPCM(RiskCapPCM(EqualRiskPCM(0.40)), target_vol=0.11)),
        )


class ClassBudgetEwmaStrategy(_MonthlyStrategy):
    """As RiskCapEwmaStrategy, plus equal risk per asset class."""

    def __init__(self):
        super().__init__(
            MonthlyTsmomAlpha((12,)),
            VolFloorRiskModel(EwmaMonthlyRiskModel()),
            LeverageCapPCM(
                VolTargetPCM(
                    RiskCapPCM(AssetClassBudgetPCM(EqualRiskPCM(0.40))), target_vol=0.11
                )
            ),
        )


class RiskCapHrpStrategy(_MonthlyStrategy):
    """HRP TSMOM + trade buffer, with each contract capped at 10% of portfolio risk."""

    def __init__(self):
        super().__init__(
            MonthlyTsmomAlpha((12,)),
            VolFloorRiskModel(EwmaMonthlyRiskModel()),
            TradeBufferPCM(
                LeverageCapPCM(VolTargetPCM(RiskCapPCM(HrpPCM()), target_vol=0.11))
            ),
        )


class ContinuousEwmaStrategy(_MonthlyStrategy):
    """'EWMA vol target + floor & caps' with the continuous signal and the trade buffer."""

    def __init__(self):
        super().__init__(
            ContinuousTsmomAlpha((12,)),
            VolFloorRiskModel(EwmaMonthlyRiskModel()),
            TradeBufferPCM(LeverageCapPCM(VolTargetPCM(EqualRiskPCM(0.40), target_vol=0.11))),
        )


class DailyTsmomVolTargetStrategy(ModularStrategy):
    """The team's daily 252d TSMOM scaled to 11% vol, with vol floor, caps and trade buffer."""

    def __init__(self):
        super().__init__(
            alpha_model=DailyTsmomAlpha(lookback=252),
            risk_model=VolFloorRiskModel(
                SampleCovarianceRiskModel(periods_per_year=252, min_periods=20)
            ),
            pcm=TradeBufferPCM(
                LeverageCapPCM(VolTargetPCM(InverseVolPCM(1.0), target_vol=0.11))
            ),
            schedule=DailySchedule(),
            risk_lookback=60,
        )


class MixedSpeedStrategy(SleeveMixStrategy):
    """50/50 mix: monthly 'EWMA vol target + floor & caps' and the daily 11%-vol TSMOM."""

    def __init__(self):
        super().__init__(
            [FloorCapEwmaVolTargetStrategy(), DailyTsmomVolTargetStrategy()], [0.5, 0.5]
        )


# Registry of user strategies to be evaluated in main.py
USER_STRATEGIES: dict[str, type[Strategy]] = {
    "Daily 252d TSMOM, inverse vol": DailyTsmomStrategy,
    "TSMOM + vol floor & caps": FloorCapTsmomStrategy,
    "TSMOM + trade buffer": BufferedTsmomStrategy,
    "EWMA vol target + floor & caps": FloorCapEwmaVolTargetStrategy,
    "HRP TSMOM (floor, caps, 11% vol)": HrpTsmomStrategy,
    "HRP TSMOM + trade buffer": HrpBufferedTsmomStrategy,
    "EWMA F&C + risk caps": RiskCapEwmaStrategy,
    "EWMA F&C + risk caps + class budgets": ClassBudgetEwmaStrategy,
    "HRP + buffer + risk caps": RiskCapHrpStrategy,
    "EWMA F&C + continuous signal + buffer": ContinuousEwmaStrategy,
    "Daily TSMOM 11% vol (floor, caps, buffer)": DailyTsmomVolTargetStrategy,
    "Mix: EWMA F&C + daily TSMOM 50/50": MixedSpeedStrategy,
}
