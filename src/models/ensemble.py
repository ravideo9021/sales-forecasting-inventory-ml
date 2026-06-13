"""
Averaging ensemble of base forecasters (e.g. XGBoost + LightGBM).

Point forecast = weighted mean of members' point forecasts.
Interval = weighted mean of members' P10/P50/P90 quantiles.

Ensembling diverse-but-comparable learners is the cheapest reliable accuracy
gain on tabular retail forecasting (consistently seen in M5 / Favorita writeups).
"""

import numpy as np
from typing import Dict, Optional
from loguru import logger


class EnsembleForecaster:
    """Combine several trained models that expose ``predict`` / ``predict_quantiles``."""

    def __init__(self, models: Dict[str, object], weights: Optional[Dict[str, float]] = None):
        self.models = {k: m for k, m in models.items() if m is not None}
        self.weights = weights or {k: 1.0 for k in self.models}
        logger.info(f"Ensemble built from: {list(self.models)}")

    def _norm_weights(self, names):
        w = np.array([self.weights.get(n, 1.0) for n in names], dtype=float)
        return w / w.sum() if w.sum() else np.ones(len(names)) / len(names)

    def predict(self, X) -> np.ndarray:
        names = list(self.models)
        preds = np.column_stack([np.asarray(self.models[n].predict(X)) for n in names])
        return np.maximum(np.average(preds, axis=1, weights=self._norm_weights(names)), 0)

    def predict_quantiles(self, X) -> Optional[Dict[str, np.ndarray]]:
        names, los, mds, his = [], [], [], []
        for n, m in self.models.items():
            q = m.predict_quantiles(X) if hasattr(m, 'predict_quantiles') else None
            if q is None:
                continue
            names.append(n); los.append(q['lo']); mds.append(q['median']); his.append(q['hi'])
        if not names:
            return None
        w = self._norm_weights(names)
        lo = np.average(np.column_stack(los), axis=1, weights=w)
        md = np.average(np.column_stack(mds), axis=1, weights=w)
        hi = np.average(np.column_stack(his), axis=1, weights=w)
        arr = np.sort(np.column_stack([lo, md, hi]), axis=1)
        return {'lo': arr[:, 0], 'median': arr[:, 1], 'hi': arr[:, 2]}
