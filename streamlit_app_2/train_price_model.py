import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

CSV_PATH = "listings.csv"

def encode_property_type(s: pd.Series) -> pd.Series:
    mapping = {"apartment":1, "house":2, "studio":0}
    return s.map(mapping).fillna(1)

def main():
    df = pd.read_csv(CSV_PATH)
    df["type_id"] = encode_property_type(df["property_type"].astype(str).str.lower())
    X = df[["area_m2","rooms","lat","lon","type_id"]].values
    y = df["price"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print("MAE:", mae)
    model.meta_ = {"features": ["area_m2","rooms","lat","lon","type_id"], "mae": float(mae)}
    joblib.dump(model, "price_model.joblib")
    print("Saved price_model.joblib")

if __name__ == "__main__":
    main()
