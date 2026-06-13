"""
Rolling-origin (expanding-window) backtest for time-series models.

A single chronological 80/20 split gives one error estimate from one cutoff.
Rolling-origin evaluation re-trains at several cutoffs and scores the next
``horizon`` of dates at each — the honest way to estimate out-of-sample error
and its variance. This is standard practice in the M5 / forecasting literature.

``model_factory`` must return a fresh estimator exposing ``fit(X, y)`` and
``predict(X)`` on plain numpy arrays (e.g. a bare ``xgboost.XGBRegressor``).
"""

import numpy as np
import pandas as pd
from loguru import logger


def rolling_origin_backtest(model_factory, X, y, dates,
                            n_splits: int = 5, horizon: int = 30,
                            min_train_frac: float = 0.5) -> pd.DataFrame:
    X = np.asarray(X)
    y = np.asarray(y).astype(float)
    dates = pd.to_datetime(pd.Series(dates).reset_index(drop=True))

    uniq = np.sort(dates.unique())
    if len(uniq) < n_splits + horizon + 1:
        logger.warning("Not enough distinct dates for a rolling backtest; skipping")
        return pd.DataFrame()

    start = int(len(uniq) * min_train_frac)
    cutoffs = np.linspace(start, len(uniq) - horizon - 1, n_splits).astype(int)
    cutoffs = sorted({int(c) for c in cutoffs if 0 < c < len(uniq) - 1})

    rows = []
    for i, c in enumerate(cutoffs, 1):
        cutoff_date = uniq[c]
        test_end = uniq[min(c + horizon, len(uniq) - 1)]
        tr = (dates <= cutoff_date).values
        te = ((dates > cutoff_date) & (dates <= test_end)).values
        if tr.sum() == 0 or te.sum() == 0:
            continue
        model = model_factory()
        model.fit(X[tr], y[tr])
        pred = np.maximum(np.asarray(model.predict(X[te])), 0)
        yt = y[te]
        wmape = np.sum(np.abs(yt - pred)) / max(np.sum(np.abs(yt)), 1e-9) * 100
        mae = float(np.mean(np.abs(yt - pred)))
        rows.append({'fold': i, 'cutoff': pd.Timestamp(cutoff_date).date(),
                     'n_train': int(tr.sum()), 'n_test': int(te.sum()),
                     'wmape': round(float(wmape), 3), 'mae': round(mae, 3)})

    df = pd.DataFrame(rows)
    if not df.empty:
        logger.info(f"Rolling backtest WMAPE: {df['wmape'].mean():.3f}% "
                    f"± {df['wmape'].std():.3f} over {len(df)} folds")
    return df
