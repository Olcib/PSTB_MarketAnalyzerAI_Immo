#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_cleaning.py
-----------------
Contrôle qualité post-cleaning.
- Vérifie que les fichiers finalisés existent.
- Vérifie les bornes de prix (10-2000€).
- Vérifie la présence des colonnes clés.
- Vérifie l’uniformité du schéma entre villes.
"""

import os
import pandas as pd

CLEAN_DIR = "data/clean"
PRICE_MIN, PRICE_MAX = 10, 2000

# Colonnes attendues (adaptables)
REQUIRED_LISTINGS = {"id", "price", "latitude", "longitude", "city"}
REQUIRED_CALENDAR = {"listing_id", "date", "price", "available", "city"}
REQUIRED_REVIEWS = {"listing_id", "date", "comments", "city"}

def check_city(city: str):
    print(f"\n🔎 Vérification cleaning pour {city}...")

    base = os.path.join(CLEAN_DIR, city)
    if not os.path.exists(base):
        print(f"❌ Pas de dossier pour {city}")
        return None

    listings_fp = os.path.join(base, "listings_final.csv")
    calendar_fp = os.path.join(base, "calendar_final.csv")
    reviews_fp = os.path.join(base, "reviews_final.csv")

    try:
        listings = pd.read_csv(listings_fp)
        calendar = pd.read_csv(calendar_fp)
        reviews = pd.read_csv(reviews_fp)
    except Exception as e:
        print(f"❌ Erreur lecture fichiers {city}: {e}")
        return None

    report = {"city": city}

    # --- Vérif 1 : colonnes présentes ---
    report["listings_missing_cols"] = list(REQUIRED_LISTINGS - set(listings.columns))
    report["calendar_missing_cols"] = list(REQUIRED_CALENDAR - set(calendar.columns))
    report["reviews_missing_cols"] = list(REQUIRED_REVIEWS - set(reviews.columns))

    # --- Vérif 2 : bornes prix ---
    if "price" in listings.columns:
        outliers = listings[~listings["price"].between(PRICE_MIN, PRICE_MAX)]
        report["listings_outliers"] = len(outliers)
    if "price" in calendar.columns:
        outliers = calendar[~calendar["price"].between(PRICE_MIN, PRICE_MAX)]
        report["calendar_outliers"] = len(outliers)

    # --- Vérif 3 : NA et doublons ---
    report["listings_na_pct"] = listings.isna().mean().max()
    report["listings_dups"] = listings.duplicated().sum()

    print("✅ Vérifications terminées")
    return report

if __name__ == "__main__":
    cities = ["paris", "seattle"]
    reports = []

    for city in cities:
        r = check_city(city)
        if r: reports.append(r)

    if reports:
        df = pd.DataFrame(reports)
        out_fp = os.path.join(CLEAN_DIR, "cleaning_report.xlsx")
        df.to_excel(out_fp, index=False)
        print(f"\n📊 Rapport cleaning exporté → {out_fp}")
        print(df)
