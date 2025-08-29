import pandas as pd
import numpy as np
from pathlib import Path
import holidays

def add_holidays(df: pd.DataFrame, country="FR") -> pd.DataFrame:
    """Ajoute une colonne holiday_flag basée sur les jours fériés du pays."""
    years = df["date"].dt.year.unique()
    fr_holidays = holidays.country_holidays(country, years=years)

    df["holiday_flag"] = df["date"].apply(lambda d: 1 if d in fr_holidays else 0)
    return df

def add_weather(df: pd.DataFrame, city: str) -> pd.DataFrame:
    """Ajoute météo simulée (température et pluie) par date."""
    np.random.seed(42)
    # Température: sinus sur l’année + bruit
    days = (df["date"] - df["date"].min()).dt.days
    df["temp_avg"] = 15 + 10 * np.sin(2 * np.pi * days / 365) + np.random.normal(0, 2, len(df))
    # Pluie: proba saisonnière
    df["rain_mm"] = np.random.gamma(2, 2, len(df)) * (0.5 + 0.5 * np.cos(2 * np.pi * days / 365))
    return df

def add_events(df: pd.DataFrame, events_file: Path) -> pd.DataFrame:
    """Ajoute événements locaux (si fichier CSV dispo)."""
    if events_file.exists():
        events = pd.read_csv(events_file, parse_dates=["date"])
        df = df.merge(events, on="date", how="left")
        df["event_flag"] = df["event_name"].notna().astype(int)
    else:
        df["event_flag"] = 0
    return df

def enrich_calendar(city: str, processed_dir="data/processed", external_dir="data/external"):
    proc_path = Path(processed_dir) / city
    ext_path = Path(external_dir)
    ext_path.mkdir(parents=True, exist_ok=True)

    cal_path = proc_path / "calendar.parquet"
    if not cal_path.exists():
        raise FileNotFoundError(f"{cal_path} introuvable. Lance d'abord ingest_core.py")

    # Charger calendar
    calendar = pd.read_parquet(cal_path)

    # Ajouter holidays
    calendar = add_holidays(calendar, country="FR")

    # Ajouter météo simulée
    calendar = add_weather(calendar, city)

    # Ajouter événements si dispo
    events_file = ext_path / f"{city}_events.csv"
    calendar = add_events(calendar, events_file)

    # Sauvegarder enrichi
    out_path = proc_path / "calendar_enriched.parquet"
    calendar.to_parquet(out_path, index=False)
    print(f"✅ Calendar enrichi sauvegardé dans {out_path}")

if __name__ == "__main__":
    for c in ["seattle", "paris"]:
        try:
            enrich_calendar(c)
        except Exception as e:
            print(f"⚠️ Erreur pour {c}: {e}")
