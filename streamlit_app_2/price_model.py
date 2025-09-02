import os
import joblib
import numpy as np
from typing import Dict, Any, Tuple

DEFAULT_MODEL_PATH = "price_model.joblib"

def load_price_model() -> Tuple[object, Dict[str, Any]]:
    path = os.getenv("PRICE_MODEL_PATH", DEFAULT_MODEL_PATH)
    if os.path.exists(path):
        model = joblib.load(path)
        meta = getattr(model, "meta_", {})
        return model, meta
    class DummyModel:
        def predict(self, X):
            base_ppm2 = 3000 + 500 * X.get("type_id", 1)
            loc_factor = 1.0 + 0.1 * ((X.get("lat", 48.85) - 48.85)**2 + (X.get("lon", 2.35) - 2.35)**2)**0.5
            price = base_ppm2 * X["area_m2"] * loc_factor * (1 + 0.02 * (X.get("rooms", 2) - 2))
            return np.array([price])
    return DummyModel(), {"note": "Using DummyModel. Provide price_model.joblib for real predictions."}

def featurize(inputs: Dict[str, Any]) -> Dict[str, float]:
    mapping_type = {"apartment":1, "house":2, "studio":0}
    return {
        "area_m2": float(inputs.get("area_m2", 50)),
        "rooms": float(inputs.get("rooms", 2)),
        "lat": float(inputs.get("lat", 48.8566)),
        "lon": float(inputs.get("lon", 2.3522)),
        "type_id": float(mapping_type.get(inputs.get("property_type","apartment"), 1)),
    }
