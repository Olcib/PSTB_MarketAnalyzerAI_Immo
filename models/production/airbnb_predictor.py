
import pandas as pd
import joblib
import numpy as np

class AirbnbPredictor:
    """
    Interface de pr�diction pour les mod�les Airbnb
    """
    
    def __init__(self, model_dir="models/production/"):
        self.model_dir = model_dir
        self.price_model = joblib.load(f"{model_dir}/price_prediction_model.pkl")
        self.rating_model = joblib.load(f"{model_dir}/rating_classification_model.pkl")
        self.scaler = joblib.load(f"{model_dir}/scaler.pkl")
        self.imputer = joblib.load(f"{model_dir}/imputer.pkl")
        
        # Charger les m�tadonn�es
        import json
        with open(f"{model_dir}/model_metadata.json", "r") as f:
            self.metadata = json.load(f)
    
    def predict_price(self, features_dict):
        """
        Pr�dit le prix d'une propri�t�
        
        Args:
            features_dict: Dictionnaire avec les features requises
        
        Returns:
            float: Prix pr�dit en dollars
        """
        # Cr�er le DataFrame avec les features
        df = pd.DataFrame([features_dict])
        
        # Preprocessing identique � l'entra�nement
        X = self.preprocess_features(df)
        
        # Pr�diction
        price_pred = self.price_model.predict(X)[0]
        
        return max(15, price_pred)  # Prix minimum $15
    
    def predict_rating_category(self, features_dict):
        """
        Pr�dit si la propri�t� sera bien not�e
        
        Args:
            features_dict: Dictionnaire avec les features requises
        
        Returns:
            str: "Bien not�" ou "Mal not�"
        """
        df = pd.DataFrame([features_dict])
        X = self.preprocess_features(df)
        
        rating_pred = self.rating_model.predict(X)[0]
        probability = self.rating_model.predict_proba(X)[0][1]
        
        result = "Bien not�" if rating_pred == 1 else "Mal not�"
        return result, probability
    
    def preprocess_features(self, df):
        """
        Preprocessing des features
        """
        # Remplir les colonnes manquantes avec des valeurs par d�faut
        for col in self.metadata['features']:
            if col not in df.columns:
                df[col] = 0
        
        # R�organiser les colonnes
        df = df.reindex(columns=self.metadata['features'], fill_value=0)
        
        # Imputation et normalisation
        X_imputed = self.imputer.transform(df)
        X_scaled = self.scaler.transform(X_imputed)
        
        return X_scaled

# Exemple d'utilisation
if __name__ == "__main__":
    predictor = AirbnbPredictor()
    
    # Exemple de pr�diction
    sample_property = {
        'occupancy_rate': 65,
        'nb_amenities': 25,
        'month': 7,
        'day_of_week': 5,
        'city_encoded': 1,  # 1=Paris, 2=Seattle
        'latitude': 48.8566,
        'longitude': 2.3522
        # Autres features seront remplies automatiquement
    }
    
    price = predictor.predict_price(sample_property)
    rating, prob = predictor.predict_rating_category(sample_property)
    
    print(f"Prix pr�dit: ${price:.2f}")
    print(f"�valuation: {rating} (probabilit�: {prob:.2f})")
