"""PatchTST forecasting interface."""
from typing import Any


class PatchTSTForecaster:
    """Integration point for a PatchTST backend."""

    def __init__(self, **config: Any) -> None:
        self.config = config
        self.model: Any = None

    def fit(self, train_data: Any) -> "PatchTSTForecaster":
        """Attach a trained backend model in a future implementation."""
        self.model = train_data
        return self

    def predict(self, inputs: Any) -> Any:
        if self.model is None:
            raise RuntimeError("Call fit before predict.")
        if hasattr(self.model, "predict"):
            return self.model.predict(inputs)
        raise NotImplementedError("Configure a PatchTST backend before prediction.")
