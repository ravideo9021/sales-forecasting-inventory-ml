"""
Optional foundation-model baseline — Amazon **Chronos** (zero-shot probabilistic
time-series forecasting with a pretrained transformer).

This is the "different / modern method" benchmark. Foundation models forecast
with little or no training and give a strong, quick baseline to compare the
gradient-boosted pipeline against (and can join the ensemble).

Heavy dependencies (``torch`` + ``chronos-forecasting``) are intentionally NOT
in requirements.txt / requirements-full.txt — they would bloat installs and the
free deploy. Install on demand:

    pip install chronos-forecasting torch

All imports are guarded so importing this module never breaks the pipeline.
"""

import numpy as np
from typing import Dict
from loguru import logger


def chronos_available() -> bool:
    try:
        import torch  # noqa: F401
        import chronos  # noqa: F401
        return True
    except Exception:
        return False


def chronos_forecast(history, horizon: int = 30,
                     model_name: str = "amazon/chronos-t5-small",
                     quantiles=(0.1, 0.5, 0.9)) -> Dict[str, np.ndarray]:
    """Zero-shot forecast of a single univariate series.

    Args:
        history: 1-D array-like of past values.
        horizon: steps to forecast.
        model_name: any Chronos checkpoint on the Hugging Face hub.
        quantiles: lower / median / upper quantiles to return.

    Returns:
        {'lo','median','hi'} arrays of length ``horizon``.
    """
    try:
        import torch
        from chronos import ChronosPipeline
    except Exception as e:  # pragma: no cover - optional dependency
        raise ImportError(
            "Chronos baseline requires `pip install chronos-forecasting torch`."
        ) from e

    pipe = ChronosPipeline.from_pretrained(model_name, device_map="cpu",
                                            torch_dtype=torch.float32)
    ctx = torch.tensor(np.asarray(history, dtype="float32"))
    forecast = pipe.predict(ctx, prediction_length=horizon)  # (n_series, n_samples, horizon)
    samples = forecast[0].numpy()
    lo, md, hi = (np.quantile(samples, q, axis=0) for q in quantiles)
    logger.info(f"Chronos zero-shot forecast generated ({model_name}, h={horizon})")
    return {'lo': np.maximum(lo, 0), 'median': np.maximum(md, 0), 'hi': np.maximum(hi, 0)}
