import math

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from common import (
    AlphaModel,
    Bar,
    HistoryContext,
    PortfolioConstructionModel,
    RiskModel,
    Strategy,
)
from indicators import (
    EfficiencyRatio,
    EwmaCrossover,
    MonthlyReturn,
    OiFlow,
    YangZhangVol,
)

ALL_HISTORY = 1_000_000


class ModularStrategy(Strategy):
    def __init__(
        self,
        alpha_model: AlphaModel,
        risk_model: RiskModel,
        pcm: PortfolioConstructionModel,
        risk_lookback: int = 252,
        rebalance: str = "daily",
    ):
        if rebalance not in ("daily", "monthly"):
            raise ValueError(f"Unsupported rebalance: {rebalance}")
        self.alpha_model = alpha_model
        self.risk_model = risk_model
        self.pcm = pcm
        self.risk_lookback = risk_lookback
        self.rebalance = rebalance
        self._signals: dict[str, float] = {}

    def on_bar(
        self, bars: dict[str, Bar], context: HistoryContext
    ) -> dict[str, float] | None:
        if not bars:
            return None

        self._signals.update(self.alpha_model.update(bars, context))
        self.risk_model.update(bars)

        if self.rebalance == "monthly" and not any(
            b.is_calendar_month_end for b in bars.values()
        ):
            return None

        cov = self.risk_model.covariance(context.get_returns(self.risk_lookback))
        return self.pcm.create_targets(self._signals, cov)


# ---- 1. Alpha Layer ------------------------------------------------------------------


class TrendAlpha(AlphaModel):
    """Combines multi-scale EwmaCrossover, EfficiencyRatio, and OiFlow into [-1, 1]."""

    def __init__(
        self,
        spans: tuple[tuple[int, int], ...] = ((8, 24), (16, 48), (32, 96), (64, 192)),
        er_lookback: int = 20,
        oi_weight: float = 0.20,
    ):
        self.spans = spans
        self.er_lookback = er_lookback
        self.oi_weight = oi_weight
        self._trends: dict[str, list[EwmaCrossover]] = {}
        self._er: dict[str, EfficiencyRatio] = {}
        self._oi: dict[str, OiFlow] = {}

    def update(self, bars: dict[str, Bar], context: HistoryContext) -> dict[str, float]:
        signals: dict[str, float] = {}
        for s, bar in bars.items():
            if s not in self._trends:
                self._trends[s] = [EwmaCrossover(f, sl) for f, sl in self.spans]
                self._er[s] = EfficiencyRatio(self.er_lookback)
                self._oi[s] = OiFlow()

            trends, er, oi = self._trends[s], self._er[s], self._oi[s]
            for t in trends:
                t.update(bar)
            er.update(bar)
            oi.update(bar)

            if not (all(t.is_ready for t in trends) and er.is_ready):
                continue

            # Average sigmoid-mapped trend forecasts across scales
            raw = sum(math.tanh(t.value / 2.0) for t in trends) / len(trends)
            er_scale = 0.5 + 0.5 * math.sqrt(er.value)
            oi_scale = 1.0 + (self.oi_weight * oi.value if oi.is_ready else 0.0)
            signals[s] = max(-1.0, min(1.0, raw * er_scale * oi_scale))

        return signals


# ---- 2. Risk Layer -------------------------------------------------------------------


def nw_shrunk_correlation(
    returns: pd.DataFrame,
    asset_classes: pd.Series,
    halflife: float = 60.0,
    nw_lags: int = 2,
    shrinkage: float = 0.25,
) -> np.ndarray:
    """EWMA Newey-West correlation matrix shrunk toward AssetClass block prior."""
    r = returns.to_numpy(dtype=np.float64)
    n_obs, n_sym = r.shape
    if n_obs < 20 or n_sym <= 1:
        return np.eye(n_sym)

    obs = np.isfinite(r)
    std = np.nanstd(r, axis=0)
    z = np.where(obs, r / np.where(std > 1e-8, std, 1.0), 0.0)
    z = np.clip(z, -4.0, 4.0)

    w = (0.5 ** (np.arange(n_obs - 1, -1, -1) / halflife))[:, None] * obs
    z_mean = (w * z).sum(axis=0, keepdims=True) / np.maximum(
        w.sum(axis=0, keepdims=True), 1e-12
    )
    z = np.where(obs, z - z_mean, 0.0)
    wz = w * z

    def _lag_cov(lag: int) -> np.ndarray:
        denom = w[lag:].T @ obs[: n_obs - lag].astype(float)
        return np.where(
            denom > 1e-6,
            (wz[lag:].T @ z[: n_obs - lag]) / np.maximum(denom, 1e-12),
            0.0,
        )

    c_nw = _lag_cov(0)
    for lag in range(1, min(nw_lags + 1, n_obs)):
        weight = 1.0 - lag / (nw_lags + 1.0)
        g = _lag_cov(lag)
        c_nw += weight * (g + g.T)

    d = np.sqrt(np.maximum(np.diag(c_nw), 1e-8))
    corr = np.clip(c_nw / np.outer(d, d), -0.99, 0.99)
    np.fill_diagonal(corr, 1.0)

    # Block target by AssetClass
    classes = asset_classes.reindex(returns.columns).to_numpy()
    target = np.eye(n_sym)
    for cls in np.unique(classes):
        idx = np.where(classes == cls)[0]
        if len(idx) > 1:
            sub = corr[np.ix_(idx, idx)]
            rho = float(
                np.clip(np.nanmean(sub[~np.eye(len(idx), dtype=bool)]), 0.0, 0.85)
            )
            target[np.ix_(idx, idx)] = rho
            np.fill_diagonal(target, 1.0)

    shrunk = (1.0 - shrinkage) * corr + shrinkage * target
    eigvals, eigvecs = np.linalg.eigh(shrunk)
    psd = eigvecs @ np.diag(np.maximum(eigvals, 1e-3)) @ eigvecs.T
    d_psd = np.sqrt(np.maximum(np.diag(psd), 1e-8))
    psd /= np.outer(d_psd, d_psd)
    np.fill_diagonal(psd, 1.0)
    return psd


class YangZhangRiskModel(RiskModel):
    """Blends fast/slow YangZhangVol for diagonal D_t and Newey-West shrunk correlation for R_t."""

    def __init__(
        self,
        asset_classes: pd.Series,
        fast_hl: float = 15.0,
        slow_hl: float = 60.0,
        corr_hl: float = 60.0,
        min_days: int = 60,
        corr_Refresh: int = 5,
    ):
        self.asset_classes = asset_classes
        self.fast_hl = fast_hl
        self.slow_hl = slow_hl
        self.corr_hl = corr_hl
        self.min_days = min_days
        self.corr_refresh = corr_Refresh

        self._fast_yz: dict[str, YangZhangVol] = {}
        self._slow_yz: dict[str, YangZhangVol] = {}
        self._steps = 0
        self._cached_corr: pd.DataFrame | None = None
        self._last_corr_step = -1

    def update(self, bars: dict[str, Bar]) -> None:
        self._steps += 1
        for s, bar in bars.items():
            if s not in self._fast_yz:
                self._fast_yz[s] = YangZhangVol(self.fast_hl, self.min_days)
                self._slow_yz[s] = YangZhangVol(self.slow_hl, self.min_days)
            self._fast_yz[s].update(bar)
            self._slow_yz[s].update(bar)

    def covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        symbols = list(returns.columns)
        vols = np.full(len(symbols), np.nan)

        for i, s in enumerate(symbols):
            f, sl = self._fast_yz.get(s), self._slow_yz.get(s)
            if f is not None and sl is not None and sl.is_ready:
                vols[i] = math.sqrt(0.7 * (f.value**2) + 0.3 * (sl.value**2))

        # Fallback for standalone risk_model_check calls where update() wasn't run
        missing = np.isnan(vols)
        if missing.any():
            sub = returns.iloc[:, missing]
            vf = (
                sub.ewm(halflife=self.fast_hl, min_periods=self.min_days).var().iloc[-1]
            )
            vs = (
                sub.ewm(halflife=self.slow_hl, min_periods=self.min_days).var().iloc[-1]
            )
            vols[missing] = np.sqrt((0.7 * vf + 0.3 * vs) * 252).to_numpy()

        valid = np.isfinite(vols) & (vols > 1e-4)
        if not valid.any():
            return pd.DataFrame(np.nan, index=symbols, columns=symbols)

        if (
            self._cached_corr is None
            or self._steps == 0
            or (self._steps - self._last_corr_step >= self.corr_refresh)
        ):
            active_syms = [s for s, v in zip(symbols, valid) if v]
            corr_act = nw_shrunk_correlation(
                returns[active_syms], self.asset_classes, self.corr_hl
            )
            self._cached_corr = pd.DataFrame(
                corr_act, index=active_syms, columns=active_syms
            )
            self._last_corr_step = self._steps

        corr = (
            self._cached_corr.reindex(index=symbols, columns=symbols)
            .fillna(0.0)
            .to_numpy(copy=True)
        )
        np.fill_diagonal(corr, 1.0)
        cov = corr * np.outer(vols, vols)
        cov[~valid, :] = np.nan
        cov[:, ~valid] = np.nan
        return pd.DataFrame(cov, index=symbols, columns=symbols)


# ---- 3. Portfolio Construction Layer -------------------------------------------------


def hrp_bisection_weights(cov: np.ndarray, corr: np.ndarray) -> np.ndarray:
    """Lopez de Prado HRP recursive bisection weights for a single cluster."""
    n = cov.shape[0]
    if n == 1:
        return np.ones(1)

    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0))
    np.fill_diagonal(dist, 0.0)
    order = list(leaves_list(linkage(squareform(dist, checks=False), method="single")))

    def _cluster_var(idx: list[int]) -> float:
        c = cov[np.ix_(idx, idx)]
        inv_v = 1.0 / np.maximum(np.diag(c), 1e-10)
        w = inv_v / inv_v.sum()
        return float(max(w @ c @ w, 1e-12))

    weights = np.ones(n)
    clusters = [order]
    while clusters:
        next_clusters = []
        for cl in clusters:
            if len(cl) <= 1:
                continue
            mid = len(cl) // 2
            left, right = cl[:mid], cl[mid:]
            vl, vr = _cluster_var(left), _cluster_var(right)
            alpha = 1.0 - vl / (vl + vr)
            weights[left] *= alpha
            weights[right] *= 1.0 - alpha
            if len(left) > 1:
                next_clusters.append(left)
            if len(right) > 1:
                next_clusters.append(right)
        clusters = next_clusters

    return weights / weights.sum()


class TwoTierHrpPCM(PortfolioConstructionModel):
    """Top tier: equal risk across AssetClasses. Bottom tier: intra-class HRP bisection."""

    def __init__(
        self,
        asset_classes: pd.Series,
        target_vol: float = 0.11,
        min_assets: int = 10,
        buffer: float = 0.12,
        speed: float = 0.55,
    ):
        self.asset_classes = asset_classes
        self.target_vol = target_vol
        self.min_assets = min_assets
        self.buffer = buffer
        self.speed = speed
        self._prev: dict[str, float] = {}
        self._conv_ref = None

    def create_targets(
        self, signals: dict[str, float], cov: pd.DataFrame
    ) -> dict[str, float]:
        vol = pd.Series(np.sqrt(np.maximum(np.diag(cov), 0.0)), index=cov.index)
        live = [
            s
            for s, sig in signals.items()
            if s in vol and vol[s] > 1e-4 and pd.notna(sig)
        ]
        if len(live) < self.min_assets:
            return {}

        sub_cov = cov.loc[live, live].fillna(0.0).to_numpy()
        sub_vol = vol.loc[live].to_numpy()
        sub_sig = np.array([signals[s] for s in live])
        sub_cls = self.asset_classes.reindex(live).fillna("OTHER").to_numpy()

        sub_corr = np.clip(sub_cov / np.outer(sub_vol, sub_vol), -0.99, 0.99)
        np.fill_diagonal(sub_corr, 1.0)
        abs_cov = np.maximum(sub_cov, 0.0)

        classes = np.unique(sub_cls)
        raw_w = np.zeros(len(live))
        sq_conv = []

        for cls in classes:
            idx = np.where(sub_cls == cls)[0]
            w_hrp = hrp_bisection_weights(
                abs_cov[np.ix_(idx, idx)], sub_corr[np.ix_(idx, idx)]
            )
            cls_vol = math.sqrt(max(w_hrp @ abs_cov[np.ix_(idx, idx)] @ w_hrp, 1e-12))
            raw_w[idx] = (w_hrp / cls_vol) * sub_sig[idx] / len(classes)

            risk_w = w_hrp * sub_vol[idx]
            sq_conv.append(float(np.sum((risk_w / risk_w.sum()) * (sub_sig[idx] ** 2))))

        conviction = math.sqrt(float(np.mean(sq_conv)))
        port_vol = math.sqrt(max(float(raw_w @ sub_cov @ raw_w), 0.0))
        if conviction < 1e-4 or port_vol <= 1e-8:
            self._prev = {}
            return {}

        ref = conviction if self._conv_ref is None else self._conv_ref
        self._conv_ref = 0.998 * ref + 0.002 * conviction  # slow EMA, past data only
        mult = min(max(conviction / max(ref, 0.05), 0.5), 1.5)
        eff_target = self.target_vol * mult
        scale = eff_target / port_vol
        opt_w = raw_w * scale

        thresh = self.buffer * eff_target / math.sqrt(len(live))
        targets: dict[str, float] = {}
        for s, w_opt, v in zip(live, opt_w, sub_vol):
            w_old = self._prev.get(s, 0.0)
            if w_old != 0.0 and abs(w_opt - w_old) * v <= thresh:
                w_new = w_old
            elif w_old == 0.0:
                w_new = float(w_opt)
            else:
                w_new = (1.0 - self.speed) * w_old + self.speed * float(w_opt)

            if abs(w_new) * v > 1e-4:
                targets[s] = w_new

        self._prev = targets.copy()
        return targets


# ---- Baseline Reference Models (kept minimal for comparison) -------------------------


class MonthlyTsmomAlpha(AlphaModel):
    """Sign of the summed past monthly returns, updated on each symbol's own month end.

    With several lookbacks the signal is the average of their signs,
    e.g. (1, 3, 6, 12) gives -1, -0.5, 0, 0.5 or 1.
    """

    def __init__(self, lookbacks: tuple[int, ...] = (12,)):
        self.lookbacks = lookbacks
        self._monthly: dict[str, MonthlyReturn] = {}

    def update(self, bars: dict[str, Bar], context: HistoryContext) -> dict[str, float]:
        signals = {}
        for symbol, bar in bars.items():
            if symbol not in self._monthly:
                self._monthly[symbol] = MonthlyReturn(max(self.lookbacks))
            ind = self._monthly[symbol]
            ind.update(bar)
            if bar.is_month_end and ind.is_ready:
                rets = list(ind.returns)
                signals[symbol] = float(
                    np.mean([np.sign(sum(rets[-k:])) for k in self.lookbacks])
                )
        return signals


class StaticLongAlpha(MonthlyTsmomAlpha):
    def update(self, bars: dict[str, Bar], context: HistoryContext) -> dict[str, float]:
        signals = super().update(bars, context)
        return {s: 1.0 for s in signals}


class ExpandingMonthlyRiskModel(RiskModel):
    """Covariance of all past monthly returns (expanding window). Needs risk_lookback=ALL_HISTORY."""

    def __init__(self, min_months: int = 36):
        self.min_months = min_months

    def covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        monthly = (1.0 + returns).groupby(returns.index.to_period("M")).prod(
            min_count=1
        ) - 1.0
        return monthly.cov(min_periods=self.min_months) * 12.0


class EqualRiskPCM(PortfolioConstructionModel):
    """Each position targets `position_vol`, then 1/N across the live symbols.

    Stays flat until at least `min_assets` symbols have both a signal and a volatility.
    """

    def __init__(self, position_vol: float = 0.40, min_assets: int = 10):
        self.position_vol = position_vol
        self.min_assets = min_assets

    def create_targets(
        self, signals: dict[str, float], cov: pd.DataFrame
    ) -> dict[str, float]:
        vol = pd.Series(np.sqrt(np.diag(cov)), index=cov.index)
        live = {s: x for s, x in signals.items() if s in vol.index and vol[s] > 1e-4}
        if len(live) < self.min_assets:
            return {}
        n = len(live)
        return {s: x * self.position_vol / vol[s] / n for s, x in live.items()}
