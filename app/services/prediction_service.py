import os
import requests
import numpy as np
import pandas as pd

from flask import current_app
from ..utils.preprocessing import preprocess_features

class PredictionService:
    def __init__(self):
        self.api_token = os.getenv("HF_API_TOKEN")
        self.endpoint = os.getenv("HF_MODEL_ENDPOINT")

        if not self.api_token or not self.endpoint:
            raise RuntimeError("Hugging Face API configuration missing")

        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }

    def predict(self, df: pd.DataFrame) -> dict:
        # Preprocess exactly as before
        df_processed = preprocess_features(df)

        payload = {
            "inputs": df_processed.to_dict(orient="records")
        }

        response = requests.post(
            self.endpoint,
            headers=self.headers,
            json=payload,
            timeout=30
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Hugging Face inference failed: {response.text}"
            )

        result = response.json()

        """
        Expected HF response format (example):
        [
          {
            "label": 2,
            "probabilities": [0.05, 0.15, 0.60, 0.20]
          }
        ]
        """

        output = result[0]
        pred_class = output["label"]
        pred_proba = np.array(output["probabilities"]) * 100

        risk_map = {
            0: "Low",
            1: "Medium",
            2: "High",
            3: "Very High"
        }

        return {
            "risk_level": risk_map[pred_class],
            "probabilities": {
                "Low": round(float(pred_proba[0]), 2),
                "Medium": round(float(pred_proba[1]), 2),
                "High": round(float(pred_proba[2]), 2),
                "Very High": round(float(pred_proba[3]), 2)
            }
        }
