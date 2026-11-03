import math

from common import Bar, Indicator


class OIVolumeAnchor(Indicator):
    """OI and Volume activity-weighted adaptive moving average of typical price (OHLC/4)."""

    def __init__(self, span: int = 126):
        if span <= 1:
            raise ValueError(f"span {span} must be > 1")
        self.span = span
        self.alpha = 2.0 / (span + 1.0)
        self._anchor_px: float | None = None
        self._ewm_activity: float | None = None
        self._count: int = 0

    @property
    def is_ready(self) -> bool:
        return self._count >= self.span and self._anchor_px is not None

    @property
    def value(self) -> float:
        return float(self._anchor_px) if self._anchor_px is not None else 0.0

    def update(self, bar: Bar) -> None:
        c = bar.close
        if math.isnan(c) or c <= 0.0:
            return

        o = bar.open if (not math.isnan(bar.open) and bar.open > 0.0) else c
        h = bar.high if (not math.isnan(bar.high) and bar.high > 0.0) else max(o, c)
        l = bar.low if (not math.isnan(bar.low) and bar.low > 0.0) else min(o, c)
        h = max(h, l, o, c)
        l = min(l, o, c)

        vol = max(bar.volume, 0.0) if not math.isnan(bar.volume) else 0.0
        oi = max(bar.open_interest, 0.0) if not math.isnan(bar.open_interest) else 0.0

        raw_act = math.log1p(vol) * math.log1p(oi)
        if raw_act <= 0.0:
            rel_act = 1.0
        elif self._ewm_activity is None or self._ewm_activity <= 0.0:
            self._ewm_activity = raw_act
            rel_act = 1.0
        else:
            rel_act = min(max(raw_act / self._ewm_activity, 0.5), 2.0)
            self._ewm_activity = (
                1.0 - self.alpha
            ) * self._ewm_activity + self.alpha * raw_act

        typical_px = 0.25 * (o + h + l + c)
        eff_alpha = min(self.alpha * rel_act, 0.5)
        if self._anchor_px is None:
            self._anchor_px = typical_px
        else:
            self._anchor_px = (
                1.0 - eff_alpha
            ) * self._anchor_px + eff_alpha * typical_px

        self._count += 1
