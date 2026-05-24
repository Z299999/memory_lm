"""yfinance-backed market streams for exp0522."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re

import numpy as np
import torch


@dataclass(frozen=True)
class MarketSeries:
    ticker: str
    price_column: str
    series_kind: str
    normalize: str
    train_values: np.ndarray
    test_values: np.ndarray
    train_dates: tuple[str, ...]
    test_dates: tuple[str, ...]
    train_mean: float
    train_std: float
    cache_path: Path

    @property
    def train_length(self) -> int:
        return int(self.train_values.shape[0])

    @property
    def test_length(self) -> int:
        return int(self.test_values.shape[0])


def experiment_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_market_cache_dir(market_cache_dir: str) -> Path:
    raw = Path(market_cache_dir).expanduser()
    if raw.is_absolute():
        return raw
    return experiment_root() / raw


def _safe_ticker_name(ticker: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", ticker.strip().upper())
    if not cleaned:
        raise ValueError("ticker must be a non-empty string.")
    return cleaned


def _cache_path(*, ticker: str, market_cache_dir: str) -> Path:
    return resolve_market_cache_dir(market_cache_dir) / f"{_safe_ticker_name(ticker)}_1d.csv"


def ensure_yfinance_csv(*, ticker: str, market_cache_dir: str) -> Path:
    """Return a cached daily OHLCV CSV, downloading with yfinance if needed."""
    path = _cache_path(ticker=ticker, market_cache_dir=market_cache_dir)
    if path.exists():
        return path

    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - depends on user env
        raise ImportError(
            "yfinance is required for target_kind='yfinance_series'. "
            "Install it with: python3 -m pip install yfinance pandas"
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    data = yf.download(
        _safe_ticker_name(ticker),
        period="max",
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if data is None or data.empty:
        raise ValueError(f"yfinance returned no daily data for ticker {ticker!r}.")
    if isinstance(data.columns, object) and hasattr(data.columns, "nlevels") and data.columns.nlevels > 1:
        data.columns = [str(col[0]) for col in data.columns]
    data = data.reset_index()
    data.to_csv(path, index=False)
    return path


@lru_cache(maxsize=16)
def load_market_series(
    *,
    ticker: str,
    price_column: str,
    series_kind: str,
    normalize: str,
    test_days: int,
    market_cache_dir: str,
) -> MarketSeries:
    """Load a cached/downloaded market stream and split train/test."""
    if test_days <= 0:
        raise ValueError("test_days must be positive.")
    if series_kind not in {"price", "log_return", "simple_return", "cumulative_log_return"}:
        raise ValueError(
            "series_kind must be 'price', 'log_return', 'simple_return', or 'cumulative_log_return'."
        )
    if normalize != "train_zscore":
        raise ValueError("Only normalize='train_zscore' is supported for yfinance_series.")

    csv_path = ensure_yfinance_csv(ticker=ticker, market_cache_dir=market_cache_dir)
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - depends on user env
        raise ImportError(
            "pandas is required for target_kind='yfinance_series'. "
            "Install it with: python3 -m pip install pandas"
        ) from exc

    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise ValueError(f"Cached market CSV is empty: {csv_path}")
    if price_column not in frame.columns:
        raise ValueError(
            f"Cached market CSV {csv_path} does not contain price_column={price_column!r}. "
            f"Available columns: {list(frame.columns)}"
        )

    date_col = "Date" if "Date" in frame.columns else frame.columns[0]
    prices = frame[price_column].to_numpy(dtype=np.float64)
    dates = tuple(str(value) for value in frame[date_col].tolist())
    finite_mask = np.isfinite(prices)
    prices = prices[finite_mask]
    dates = tuple(date for date, keep in zip(dates, finite_mask) if bool(keep))
    if series_kind == "price":
        values = prices
        value_dates = dates
    elif series_kind == "cumulative_log_return":
        if np.any(prices <= 0.0):
            raise ValueError(f"{series_kind} requires strictly positive {price_column!r} values.")
        values = np.log(prices / prices[0])
        value_dates = dates
    else:
        if np.any(prices <= 0.0):
            raise ValueError(f"{series_kind} requires strictly positive {price_column!r} values.")
        if series_kind == "log_return":
            values = np.diff(np.log(prices))
        else:
            values = prices[1:] / prices[:-1] - 1.0
        value_dates = dates[1:]

    min_rows = test_days + 2
    if values.shape[0] < min_rows:
        raise ValueError(
            f"Not enough finite {series_kind} rows from {price_column!r} for {ticker!r}: "
            f"got {values.shape[0]}, need at least {min_rows}."
        )

    train_raw = values[:-test_days]
    test_raw = values[-test_days:]
    train_dates = value_dates[:-test_days]
    test_dates = value_dates[-test_days:]
    train_mean = float(np.mean(train_raw))
    train_std = float(np.std(train_raw))
    if not np.isfinite(train_std) or train_std <= 0.0:
        raise ValueError("Training market series standard deviation must be finite and positive.")

    train_values = ((train_raw - train_mean) / train_std).astype(np.float32)
    test_values = ((test_raw - train_mean) / train_std).astype(np.float32)
    return MarketSeries(
        ticker=_safe_ticker_name(ticker),
        price_column=str(price_column),
        series_kind=str(series_kind),
        normalize=str(normalize),
        train_values=train_values,
        test_values=test_values,
        train_dates=train_dates,
        test_dates=test_dates,
        train_mean=train_mean,
        train_std=train_std,
        cache_path=csv_path,
    )


def build_market_rollout_targets(
    *,
    num_steps: int,
    device: torch.device,
    start_step: int,
    split: str,
    ticker: str,
    price_column: str,
    series_kind: str,
    normalize: str,
    test_days: int,
    market_cache_dir: str,
) -> dict[str, torch.Tensor]:
    """Return a finite market target slice compatible with build_rollout_targets."""
    series = load_market_series(
        ticker=ticker,
        price_column=price_column,
        series_kind=series_kind,
        normalize=normalize,
        test_days=test_days,
        market_cache_dir=market_cache_dir,
    )
    if split == "train":
        values = series.train_values
        global_offset = 0
    elif split == "test":
        values = series.test_values
        global_offset = series.train_length
    else:
        raise ValueError("split must be either 'train' or 'test'.")

    if start_step < 0:
        raise ValueError("start_step must be non-negative.")
    end_step = start_step + num_steps
    if end_step > values.shape[0]:
        raise ValueError(
            f"Requested {split} market slice [{start_step}, {end_step}) but "
            f"only {values.shape[0]} steps are available."
        )

    target = torch.as_tensor(values[start_step:end_step], dtype=torch.float32, device=device).unsqueeze(1)
    phase = torch.arange(
        global_offset + start_step,
        global_offset + end_step,
        dtype=torch.float32,
        device=device,
    )
    return {
        "phase": phase,
        "target_y": target,
        "train_target": target,
        "global_start_step": torch.tensor(global_offset + start_step, dtype=torch.int64),
    }
