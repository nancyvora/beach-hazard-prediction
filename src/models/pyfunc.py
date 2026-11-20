"""Single registered inference object owns preprocessing and calibration."""

import mlflow.pyfunc
import numpy as np

from src.models.calibration import logit


class RiskModel(mlflow.pyfunc.PythonModel):
    def __init__(self, pipeline, calibrator, columns):
        self.pipeline = pipeline
        self.calibrator = calibrator
        self.columns = list(columns)

    def predict(self, context, model_input, params=None):
        frame = model_input[self.columns]
        positive = int(np.flatnonzero(self.pipeline.classes_ == 1)[0])
        scores = self.pipeline.predict_proba(frame)[:, positive]
        if self.calibrator is not None:
            scores = self.calibrator.predict_proba(logit(scores))[:, 1]
        return np.asarray(scores, dtype=float)
