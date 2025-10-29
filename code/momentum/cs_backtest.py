
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple
import numpy as np
import pandas as pd

# -----------------------------
# Utility helpers
# -----------------------------

def _to_dt(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    if not np.issubdtype(df[col].dtype, np.datetime64):
        df = df.copy()
        df[col] = pd.to_datetime(df[col])
    return df

def _ensure_sorted(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["date","TICKER"]).reset_index(drop=True)

def winsorize_s(col: pd.Series, lower=0.01, upper=0.99):
    lo, hi = col.quantile([lower, upper])
    return col.clip(lo, hi)

def zscore(col: pd.Series):
    s = col.astype(float)
    return (s - s.mean())/s.std(ddof=0) if s.std(ddof=0) > 0 else s*0.0

def top_bottom_weights(sig: pd.Series, q: float = 0.2) -> pd.Series:
    # Long top q, short bottom q, dollar-neutral, equal weight within sides
    sig = sig.dropna()
    if sig.empty:
        return pd.Series(dtype=float)
    ql = sig.quantile(q)
    qh = sig.quantile(1-q)
    long = sig[sig >= qh].index
    short = sig[sig <= ql].index
    w = pd.Series(0.0, index=sig.index, dtype=float)
    if len(long) > 0:
        w.loc[long] =  1.0/len(long)
    if len(short) > 0:
        w.loc[short] = -1.0/len(short)
    # Dollar neutral by construction
    return w

def linear_weights_from_zscore(sig: pd.Series) -> pd.Series:
    # Map z-scored signal to weights, rescale to dollar-neutral |w| sum = 1
    s = zscore(sig.dropna())
    if s.empty or s.abs().sum()==0:
        return pd.Series(0.0, index=s.index, dtype=float)
    w = s / s.abs().sum()
    return w

# -----------------------------
# Backtest
# -----------------------------

SignalFn = Callable[[pd.DataFrame], pd.Series]
WeightFn = Callable[[pd.Series], pd.Series]

@dataclass
class CSBacktestConfig:
    rebalance_freq: str = "W-FRI"       # pandas offset alias
    transaction_cost_bps: float = 0.0   # per-dollar one-way
    min_names: int = 20                 # skip rebalances if fewer names
    use_close_to_close: bool = True     # if True, signal at t applied to [t+1, next_reb]
    weight_scheme: str = "top_bottom"   # "top_bottom" or "zlinear"
    quantile: float = 0.2               # only for top_bottom
    cap_turnover: Optional[float] = None  # e.g., 1.5 -> cap total |Δw| at 150%
    leverage: float = 1.0               # scale all weights

class CrossSectionalBacktester:
    def __init__(
        self,
        data: pd.DataFrame,
        signal_fn: SignalFn,
        config: CSBacktestConfig = CSBacktestConfig(),
        signal_col_name: str = "signal"
    ):
        # 'data' must be a daily panel with columns ['date','TICKER','RET', ...].
        # 'signal_fn' should return a Series indexed by ['date','TICKER'] with the signal.
        self.raw = _ensure_sorted(_to_dt(data.copy(), "date"))
        self.signal_fn = signal_fn
        self.cfg = config
        self.signal_col_name = signal_col_name
        self._prepared: Optional[pd.DataFrame] = None

    def _prepare(self) -> pd.DataFrame:
        df = self.raw.copy()
        if "RET" not in df.columns:
            raise ValueError("Input data must contain a 'RET' column with simple returns.")
        # compute signal via user function
        sig = self.signal_fn(df)  # expects MultiIndex (date, TICKER)
        if isinstance(sig, pd.Series):
            sig.name = self.signal_col_name
            sig = sig.reset_index()
        elif isinstance(sig, pd.DataFrame) and self.signal_col_name in sig.columns:
            pass
        else:
            raise ValueError("signal_fn must return a Series or DataFrame with signal column.")
        # merge
        df = df.merge(sig, on=["date","TICKER"], how="left")
        return df

    def _rebalance_dates(self, df: pd.DataFrame) -> pd.DatetimeIndex:
        # Use last available date in each week based on rebalance_freq
        by = df.groupby("date").size().index
        # Resample on a dummy series to get calendar then align to existing dates
        cal = pd.Series(1, index=by).resample(self.cfg.rebalance_freq).last().index
        # Map each calendar date to the last trading date <= that calendar point
        rebs = []
        dates = by.sort_values()
        i = 0
        last = None
        for d in cal:
            while i < len(dates) and dates[i] <= d:
                last = dates[i]
                i += 1
            if last is not None:
                rebs.append(last)
        rebs = pd.DatetimeIndex(pd.unique(rebs))
        return rebs

    def _choose_weight_fn(self) -> WeightFn:
        if self.cfg.weight_scheme == "top_bottom":
            return lambda s: top_bottom_weights(s, q=self.cfg.quantile)
        elif self.cfg.weight_scheme == "zlinear":
            return linear_weights_from_zscore
        else:
            raise ValueError("Unknown weight_scheme")

    def run(self) -> Dict[str, pd.DataFrame]:
        df = self._prepare()
        wfn = self._choose_weight_fn()
        rebs = self._rebalance_dates(df)

        # Build daily panel for quick lookup
        df = df.set_index(["date","TICKER"]).sort_index()

        # For cost turnover accounting
        prev_w = pd.Series(dtype=float)

        daily_rows = []  # collect dicts of results

        for t_idx, t in enumerate(rebs):
            # signal snapshot at t (all tickers at t)
            s_t = (
                df.xs(t, level="date")[self.signal_col_name].dropna()
                if (t, slice(None)) in df.index
                else pd.Series(dtype=float)
            )
            if s_t.empty or len(s_t.index.unique()) < self.cfg.min_names:
                continue

            # build weights for the next period
            w_t = wfn(s_t)
            if self.cfg.leverage != 1.0:
                w_t = w_t * self.cfg.leverage

            # turnover (one-way) at this rebalance: sum |Δw|
            aligned_prev = prev_w.reindex(w_t.index).fillna(0.0)
            delta = (w_t - aligned_prev).abs().sum()
            if (self.cfg.cap_turnover is not None) and (delta > self.cfg.cap_turnover):
                scale = self.cfg.cap_turnover / delta
                w_t = aligned_prev + (w_t - aligned_prev) * scale
                delta = self.cfg.cap_turnover

            # Determine holding window: (t, t_next]
            t_next = rebs[min(t_idx+1, len(rebs)-1)]
            if t_next == t:
                continue
            window = df.loc[(slice(t, t_next), w_t.index), "RET"].unstack("TICKER")
            # If use_close_to_close, start from t+1
            if self.cfg.use_close_to_close and (len(window.index) > 1):
                window = window.iloc[1:]

            # daily pnl
            cost_applied = False
            for i, (day, retrow) in enumerate(window.iterrows()):
                r = retrow.fillna(0.0)
                pnl = float((w_t.reindex(r.index).fillna(0.0) * r).sum())
                # apply one-time transaction cost on first day in window
                if not cost_applied and self.cfg.transaction_cost_bps:
                    cost = self.cfg.transaction_cost_bps/1e4 * delta
                    pnl -= cost
                    cost_applied = True
                daily_rows.append({
                    "date": day,
                    "ret": pnl,
                    "turnover": 0.0 if i>0 else float(delta),
                    "gross_leverage": float(w_t.abs().sum())
                })

            prev_w = w_t

        if len(daily_rows) == 0:
            raise RuntimeError("No PnL produced. Check min_names, rebalance calendar, or signal coverage.")

        out = pd.DataFrame(daily_rows).groupby("date").sum().sort_index()
        out["cumret"] = (1.0 + out["ret"]).cumprod()
        return {"performance": out}
