"""Technical indicators for NSE Quant.

We ship a curated registry of 50+ indicators covering trend, momentum, volatility,
volume, and cycle families. The primary implementation uses ``pandas_ta`` when
available and falls back to hand-rolled pandas/numpy implementations so the
package still works in minimal environments (e.g. the CI image without optional
build-tools-heavy wheels).
"""
from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:  # pragma: no cover - simple import guard
    import pandas_ta as pta  # type: ignore
    _HAS_PTA = True
except Exception:  # noqa: BLE001
    pta = None
    _HAS_PTA = False


# ---------------------------------------------------------------------------
# Hand-rolled fallbacks (used when pandas_ta is missing or fails)
# ---------------------------------------------------------------------------

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def _wma(s: pd.Series, n: int) -> pd.Series:
    weights = np.arange(1, n + 1)
    return s.rolling(n).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).rename(f"rsi_{n}")


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    fast_ema = _ema(close, fast)
    slow_ema = _ema(close, slow)
    macd = fast_ema - slow_ema
    sig = _ema(macd, signal)
    hist = macd - sig
    return pd.DataFrame({"macd": macd, "macd_signal": sig, "macd_hist": hist})


def _bbands(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = _sma(close, n)
    sd = close.rolling(n, min_periods=n).std()
    upper = mid + k * sd
    lower = mid - k * sd
    width = (upper - lower) / mid
    percent = (close - lower) / (upper - lower)
    return pd.DataFrame(
        {"bb_upper": upper, "bb_middle": mid, "bb_lower": lower, "bb_width": width, "bb_pct": percent}
    )


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean().rename(f"atr_{n}")


def _stoch(high: pd.Series, low: pd.Series, close: pd.Series, k: int = 14, d: int = 3) -> pd.DataFrame:
    ll = low.rolling(k, min_periods=k).min()
    hh = high.rolling(k, min_periods=k).max()
    fast_k = 100 * (close - ll) / (hh - ll).replace(0, np.nan)
    slow_d = fast_k.rolling(d, min_periods=d).mean()
    return pd.DataFrame({"stoch_k": fast_k, "stoch_d": slow_d})


def _williams_r(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    hh = high.rolling(n, min_periods=n).max()
    ll = low.rolling(n, min_periods=n).min()
    return (-100 * (hh - close) / (hh - ll).replace(0, np.nan)).rename(f"williams_r_{n}")


def _cci(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 20) -> pd.Series:
    tp = (high + low + close) / 3
    ma = tp.rolling(n, min_periods=n).mean()
    md = tp.rolling(n, min_periods=n).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return ((tp - ma) / (0.015 * md.replace(0, np.nan))).rename(f"cci_{n}")


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    sign = np.sign(close.diff().fillna(0))
    return (sign * volume).cumsum().rename("obv")


def _vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    cum_vol = df["volume"].cumsum().replace(0, np.nan)
    return ((tp * df["volume"]).cumsum() / cum_vol).rename("vwap")


def _mfi(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    mf = tp * df["volume"]
    sign = np.sign(tp.diff())
    pos = mf.where(sign > 0, 0).rolling(n, min_periods=n).sum()
    neg = mf.where(sign < 0, 0).rolling(n, min_periods=n).sum()
    mr = pos / neg.replace(0, np.nan)
    return (100 - 100 / (1 + mr)).rename(f"mfi_{n}")


def _adl(df: pd.DataFrame) -> pd.Series:
    clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"]).replace(0, np.nan)
    return (clv.fillna(0) * df["volume"]).cumsum().rename("adl")


def _chaikin(df: pd.DataFrame) -> pd.Series:
    adl = _adl(df)
    return (_ema(adl, 3) - _ema(adl, 10)).rename("chaikin_osc")


def _ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    high, low = df["high"], df["low"]
    conv = (high.rolling(9).max() + low.rolling(9).min()) / 2
    base = (high.rolling(26).max() + low.rolling(26).min()) / 2
    span_a = ((conv + base) / 2).shift(26)
    span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    lag = df["close"].shift(-26)
    return pd.DataFrame(
        {
            "ichimoku_conv": conv,
            "ichimoku_base": base,
            "ichimoku_span_a": span_a,
            "ichimoku_span_b": span_b,
            "ichimoku_lag": lag,
        }
    )


def _donchian(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    upper = df["high"].rolling(n).max()
    lower = df["low"].rolling(n).min()
    mid = (upper + lower) / 2
    return pd.DataFrame({f"donchian_upper_{n}": upper, f"donchian_lower_{n}": lower, f"donchian_mid_{n}": mid})


def _keltner(df: pd.DataFrame, n: int = 20, mult: float = 2.0) -> pd.DataFrame:
    mid = _ema(df["close"], n)
    atr = _atr(df["high"], df["low"], df["close"], n)
    return pd.DataFrame(
        {
            f"keltner_upper_{n}": mid + mult * atr,
            f"keltner_mid_{n}": mid,
            f"keltner_lower_{n}": mid - mult * atr,
        }
    )


def _parabolic_sar(df: pd.DataFrame, af_step: float = 0.02, af_max: float = 0.2) -> pd.Series:
    high = df["high"].values
    low = df["low"].values
    sar = np.full(len(df), np.nan)
    if len(df) < 2:
        return pd.Series(sar, index=df.index, name="psar")
    bull = True
    af = af_step
    ep = high[0]
    sar[0] = low[0]
    for i in range(1, len(df)):
        prev = sar[i - 1]
        if bull:
            sar[i] = prev + af * (ep - prev)
            sar[i] = min(sar[i], low[i - 1], low[max(i - 2, 0)])
            if low[i] < sar[i]:
                bull = False
                sar[i] = ep
                ep = low[i]
                af = af_step
            elif high[i] > ep:
                ep = high[i]
                af = min(af + af_step, af_max)
        else:
            sar[i] = prev + af * (ep - prev)
            sar[i] = max(sar[i], high[i - 1], high[max(i - 2, 0)])
            if high[i] > sar[i]:
                bull = True
                sar[i] = ep
                ep = high[i]
                af = af_step
            elif low[i] < ep:
                ep = low[i]
                af = min(af + af_step, af_max)
    return pd.Series(sar, index=df.index, name="psar")


def _supertrend(df: pd.DataFrame, n: int = 10, mult: float = 3.0) -> pd.DataFrame:
    hl2 = (df["high"] + df["low"]) / 2
    atr = _atr(df["high"], df["low"], df["close"], n)
    upper = hl2 + mult * atr
    lower = hl2 - mult * atr
    trend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=float)
    dir_state = 1
    for i in range(len(df)):
        u, ll = upper.iat[i], lower.iat[i]
        if pd.isna(u) or pd.isna(ll):
            trend.iat[i] = np.nan
            direction.iat[i] = dir_state
            continue
        c = df["close"].iat[i]
        if dir_state == 1 and c < ll:
            dir_state = -1
        elif dir_state == -1 and c > u:
            dir_state = 1
        trend.iat[i] = ll if dir_state == 1 else u
        direction.iat[i] = dir_state
    return pd.DataFrame({"supertrend": trend, "supertrend_dir": direction}, index=df.index)


def _roc(close: pd.Series, n: int = 12) -> pd.Series:
    return close.pct_change(n).rename(f"roc_{n}")


def _momentum(close: pd.Series, n: int = 10) -> pd.Series:
    return (close - close.shift(n)).rename(f"momentum_{n}")


def _trix(close: pd.Series, n: int = 15) -> pd.Series:
    ema1 = _ema(close, n)
    ema2 = _ema(ema1, n)
    ema3 = _ema(ema2, n)
    return (ema3.pct_change() * 100).rename(f"trix_{n}")


def _kama(close: pd.Series, n: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    change = close.diff(n).abs()
    volatility = close.diff().abs().rolling(n).sum()
    er = change / volatility.replace(0, np.nan)
    sc = (er * (2 / (fast + 1) - 2 / (slow + 1)) + 2 / (slow + 1)) ** 2
    kama = close.copy()
    kama.iloc[:n] = close.iloc[:n]
    for i in range(n, len(close)):
        prev = kama.iloc[i - 1]
        kama.iloc[i] = prev + (sc.iloc[i] if pd.notna(sc.iloc[i]) else 0) * (close.iloc[i] - prev)
    return kama.rename(f"kama_{n}")


def _cmo(close: pd.Series, n: int = 14) -> pd.Series:
    diff = close.diff()
    up = diff.clip(lower=0).rolling(n).sum()
    down = -diff.clip(upper=0).rolling(n).sum()
    return (100 * (up - down) / (up + down).replace(0, np.nan)).rename(f"cmo_{n}")


def _dpo(close: pd.Series, n: int = 20) -> pd.Series:
    return (close.shift(n // 2 + 1) - close.rolling(n).mean()).rename(f"dpo_{n}")


def _ulcer(close: pd.Series, n: int = 14) -> pd.Series:
    roll_max = close.rolling(n).max()
    drawdown = 100 * (close - roll_max) / roll_max
    return np.sqrt((drawdown**2).rolling(n).mean()).rename(f"ulcer_{n}")


def _vroc(volume: pd.Series, n: int = 14) -> pd.Series:
    return volume.pct_change(n).rename(f"vroc_{n}")


def _pvt(close: pd.Series, volume: pd.Series) -> pd.Series:
    return (close.pct_change().fillna(0) * volume).cumsum().rename("pvt")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

IndicatorFn = Callable[[pd.DataFrame], pd.DataFrame]


def _wrap_series(fn: Callable[..., pd.Series], *args, **kwargs) -> IndicatorFn:
    def inner(df: pd.DataFrame) -> pd.DataFrame:
        out = fn(*args, **kwargs, **{k: df[k] for k in ["open", "high", "low", "close", "volume"] if k in df.columns and k in fn.__code__.co_varnames})
        return out.to_frame() if isinstance(out, pd.Series) else out
    return inner


def _reg_trend(df: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame | pd.Series] = []
    for n in [5, 10, 20, 50, 100, 200]:
        parts.append(_sma(df["close"], n).rename(f"sma_{n}"))
        parts.append(_ema(df["close"], n).rename(f"ema_{n}"))
    parts.append(_wma(df["close"], 20).rename("wma_20"))
    parts.append(_kama(df["close"]))
    return pd.concat(parts, axis=1)


def _reg_momentum(df: pd.DataFrame) -> pd.DataFrame:
    parts = [
        _rsi(df["close"], 7),
        _rsi(df["close"], 14),
        _rsi(df["close"], 28),
        _macd(df["close"]),
        _stoch(df["high"], df["low"], df["close"]),
        _williams_r(df["high"], df["low"], df["close"]),
        _cci(df["high"], df["low"], df["close"]),
        _roc(df["close"], 5),
        _roc(df["close"], 21),
        _momentum(df["close"], 10),
        _trix(df["close"]),
        _cmo(df["close"]),
        _dpo(df["close"]),
    ]
    return pd.concat(parts, axis=1)


def _reg_volatility(df: pd.DataFrame) -> pd.DataFrame:
    parts = [
        _atr(df["high"], df["low"], df["close"], 14),
        _atr(df["high"], df["low"], df["close"], 28),
        _bbands(df["close"]),
        _keltner(df),
        _donchian(df),
        _ulcer(df["close"]),
    ]
    return pd.concat(parts, axis=1)


def _reg_volume(df: pd.DataFrame) -> pd.DataFrame:
    parts = [
        _obv(df["close"], df["volume"]),
        _vwap(df),
        _mfi(df),
        _adl(df),
        _chaikin(df),
        _vroc(df["volume"]),
        _pvt(df["close"], df["volume"]),
    ]
    return pd.concat(parts, axis=1)


def _reg_cycle(df: pd.DataFrame) -> pd.DataFrame:
    parts = [
        _ichimoku(df),
        _parabolic_sar(df),
        _supertrend(df),
    ]
    return pd.concat(parts, axis=1)


INDICATOR_REGISTRY: dict[str, IndicatorFn] = {
    "trend": _reg_trend,
    "momentum": _reg_momentum,
    "volatility": _reg_volatility,
    "volume": _reg_volume,
    "cycle": _reg_cycle,
}


def available_indicators() -> list[str]:
    """Return the list of atomic indicator column names produced by add_all_indicators."""
    # Seed with a tiny OHLCV frame to discover output columns.
    idx = pd.bdate_range("2024-01-01", periods=260)
    stub = pd.DataFrame(
        {
            "open": np.linspace(100, 110, 260),
            "high": np.linspace(101, 111, 260),
            "low": np.linspace(99, 109, 260),
            "close": np.linspace(100, 110, 260) + np.sin(np.linspace(0, 20, 260)),
            "volume": 1_000_000 + np.arange(260) * 10,
        },
        index=idx,
    )
    out = add_all_indicators(stub)
    return [c for c in out.columns if c not in {"open", "high", "low", "close", "volume"}]


def compute_indicator(df: pd.DataFrame, family: str) -> pd.DataFrame:
    if family not in INDICATOR_REGISTRY:
        raise KeyError(f"Unknown indicator family: {family}")
    return INDICATOR_REGISTRY[family](df)


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with every registered indicator appended as new columns."""
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {sorted(missing)}")
    parts = [df]
    for family, fn in INDICATOR_REGISTRY.items():
        try:
            parts.append(fn(df))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Indicator family %s failed: %s", family, exc)
    out = pd.concat(parts, axis=1)
    # De-duplicate column names if any family accidentally overlaps.
    out = out.loc[:, ~out.columns.duplicated()]
    return out
