from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class BadMagnitudeScaler(BaseEstimator, TransformerMixin):
    """Deliberately distort feature magnitudes to show why scaling matters."""

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("BadMagnitudeScaler expects a 2D array.")

        self.n_features_in_ = X.shape[1]
        midpoint = (self.n_features_in_ - 1) / 2
        self.scale_factors_ = np.power(
            10.0,
            np.arange(self.n_features_in_, dtype=float) - midpoint,
        )
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("BadMagnitudeScaler expects a 2D array.")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "BadMagnitudeScaler saw a different number of features at transform time."
            )

        return X * self.scale_factors_

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        return np.asarray(input_features, dtype=object)
