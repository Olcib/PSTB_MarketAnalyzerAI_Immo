#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clean_and_eda_light.py
----------------------
Nettoyage + EDA léger des datasets Airbnb (Paris, Seattle).
- Lecture tolérante aux lignes corrompues.
- Nettoyage des prix, dates, disponibilité.
- Exporte fichiers nettoyés allégés.
- Génère un rapport Excel bilingue (EN + FR).
"""

import os
import glob
import pandas as pd

PROC_DIR = "data/processed"
OUT_DIR = "data/clean_light"
os.makedirs(OUT_DIR, exist_ok=True)

PRICE_MIN, PRICE_MAX = 10, 2000


# -----------------------------
# Helpers
# -----------------------------
def pick_file(city_dir, patterns):
    """Retourne le fichier le plus récent qui matche un pattern"""
    for pat in patterns:
        matches = glob.glob(os.path.join(city_dir, pat))
        if matches:
            matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return matches[0]
    return None


def safe_read_csv(path):
    """Lecture robuste avec skip lignes corrompues"""
    try:
        df = pd.read_csv(
            path,
            sep=",",
            quotechar='"',
            low_memory=False,
            on_bad_lines="skip"  # saute les lignes invalides
        )
        print(f"📥 Lecture OK: {path} → {df.shape[0]} lignes, {df.shape[1]} colonnes")
        return df
    except Exception as e:
        print(f"❌ Erreur lecture {path}: {e}")
        return pd.DataFrame()


# -----------------------------
# Cleaning d'une ville
# -----------------------------
def clean_city(city, max_rows=None):
    print(f"\n🧹 Cleaning & EDA (light) → {city}")

    city_in = os.path.join(PROC_DIR, city)
    city_out = os.path.join(OUT_DIR, city)
    os.makedirs(city_out, exist_ok=True)

    # Fichiers
    listings_fp = pick_file(city_in, ["listings_clean*.csv", "listings.csv"])
    calendar_fp = pick_file(city_in, ["calendar_clean*.csv", "calendar.csv"])
    reviews_fp = pick_file(city_in, ["reviews_clean*.csv", "reviews.csv"])

    listings = safe_read_csv(listings_fp) if listings_fp else pd.DataFrame()
    calendar = safe_read_csv(calendar_fp) if calendar_fp else pd.DataFrame()
    reviews = safe_read_csv(reviews_fp) if reviews_fp else pd.DataFrame()

    # --- Listings
    if not listings.empty and "price" in listings.columns:
        listings["price"] = (
            listings["price"].astype(str).str.replace(r"[^\d.]", "", regex=True)
        )
        listings["price"] = pd.to_numeric(listings["price"], errors="coerce")
        listings = listings[listings["price"].between(PRICE_MIN, PRICE_MAX, inclusive="neither")]

    if not listings.empty and {"latitude", "longitude"}.issubset(listings.columns):
        listings = listings.dropna(subset=["latitude", "longitude"])

    # --- Calendar
    if not calendar.empty:
        if "price" in calendar.columns:
            calendar["price"] = calendar["price"].astype(str).str.replace(r"[^\d.]", "", regex=True)
            calendar["price"] = pd.to_numeric(calendar["price"], errors="coerce")
            calendar = calendar[calendar["price"].between(PRICE_MIN, PRICE_MAX, inclusive="neither")]
        if "date" in calendar.columns:
            calendar["date"] = pd.to_datetime(calendar["date"], errors="coerce")
        if "available" in calendar.columns:
            calendar["available"] = calendar["available"].astype(str).str.lower().map(
                {"t": 1, "f": 0, "true": 1, "false": 0}
            ).fillna(0).astype(int)

    # --- Reviews
    if not reviews.empty and "date" in reviews.columns:
        reviews["date"] = pd.to_datetime(reviews["date"], errors="coerce")

    # --- Échantillonnage
    if max_rows:
        listings = listings.head(max_rows)
        if not calendar.empty:
            calendar = calendar.head(max_rows)
        if not reviews.empty:
            reviews = reviews.head(max_rows)

    # --- Sauvegarde
    listings.to_csv(os.path.join(city_out, "listings_light.csv"), index=False)
    calendar.to_csv(os.path.join(city_out, "calendar_light.csv"), index=False)
    reviews.to_csv(os.path.join(city_out, "reviews_light.csv"), index=False)

    print(f"✅ {city} → {city_out}")

    return {
        "City / Ville": city,
        "Listings shape / Taille listings": f"{listings.shape}",
        "Calendar shape / Taille calendrier": f"{calendar.shape}",
        "Reviews shape / Taille reviews": f"{reviews.shape}",
        "Calendar period / Période calendrier": f"{calendar['date'].min()} → {calendar['date'].max()}" if not calendar.empty else "vide",
        "Reviews period / Période reviews": f"{reviews['date'].min()} → {reviews['date'].max()}" if not reviews.empty else "vide",
        "NA top5 (%)": listings.isna().mean().sort_values(ascending=False).head(5).to_dict() if not listings.empty else {},
        "Duplicates / Doublons": int(listings.duplicated().sum()) if not listings.empty else 0
    }


# -----------------------------
# Main
# -----------------------------
def main():
    cities = [d for d in os.listdir(PROC_DIR) if os.path.isdir(os.path.join(PROC_DIR, d))]
    rows = []
    for city in cities:
        rows.append(clean_city(city, max_rows=8000))

    # Export rapport
    out_xlsx = os.path.join(OUT_DIR, "eda_report_bilingual.xlsx")
    df = pd.DataFrame(rows)

    # Formater NA top5
    if "NA top5 (%)" in df.columns:
        df["NA top5 (%)"] = df["NA top5 (%)"].apply(
            lambda d: ", ".join([f"{k}:{v:.1%}" for k, v in d.items()]) if isinstance(d, dict) else str(d)
        )

    df.to_excel(out_xlsx, index=False)
    print(f"\n📊 Rapport EDA exporté → {out_xlsx}")


if __name__ == "__main__":
    main()
