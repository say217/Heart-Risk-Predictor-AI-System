import joblib
import numpy as np
import pandas as pd
from flask import current_app
import os

class PredictionService:
    def __init__(self):
        self.model = None  # Will load on first use

    def _load_model(self):
        if self.model is None:
            model_path = current_app.config['MODEL_PATH']
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            self.model = joblib.load(model_path)
            print(f"Model successfully loaded from {model_path}")

    def predict(self, df: pd.DataFrame) -> dict:
        self._load_model()  # Lazy load only when needed

        from ..utils.preprocessing import preprocess_features
        df_processed = preprocess_features(df)

        pred_class = self.model.predict(df_processed)[0]
        pred_proba = self.model.predict_proba(df_processed)[0]

        risk_map = {0: "Low", 1: "Medium", 2: "High", 3: "Very High"}
        proba_percent = np.round(pred_proba * 100, 2)

        return {
            "risk_level": risk_map[pred_class],
            "probabilities": {
                "Low": float(proba_percent[0]),
                "Medium": float(proba_percent[1]),
                "High": float(proba_percent[2]),
                "Very High": float(proba_percent[3])
            }
        }