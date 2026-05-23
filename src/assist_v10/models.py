import numpy as np


class DummyModel:
    """Modelo dummy para demostración cuando no hay modelos entrenados."""

    def predict(self, X):
        return np.random.uniform(0.3, 0.85, size=len(X) if hasattr(X, '__len__') else 1)

    def predict_proba(self, X):
        proba = self.predict(X)
        return np.column_stack([1-proba, proba])
