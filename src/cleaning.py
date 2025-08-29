#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cleaning.py
-----------
Étape D — Nettoyage avancé.
- Supprime les outliers prix (<10€ ou >2000€/nuit).
- Corrige les types (dates, numériques).
- Supprime les annonces sans coordonnées.
- Uniformise les colonnes entre villes.
- Sauvegarde dans data/clean/<ville>/*_final.csv
"""

import os
import pandas as pd

PROC_DIR = "data/processed"
CLEAN_DIR = "data/clean"
os.makedirs(CLEAN_DIR, exist_ok=True)

PRICE_MIN, PRICE_MAX = 10, 2000

def clean_city(city: str):
    print(f"\n🧹 Nettoyage avancé pour {city}...")

    path = os.path.join(PROC_DIR, city)
    out_path = os.path.join(CLEAN_DIR, city)
    os.makedirs(out_path, exist_ok=True)

    try:
        listings = pd.read_csv(os.path.join(path, "listings_clean.csv"))
        calendar = pd.read_csv(os.path.join(path, "calendar_clean.csv"))
        reviews = pd.read_csv(os.path.join(path, "reviews_clean.csv"))
    except Exception as e:
        print(f"❌ Impossible de charger les fichiers pour {city}: {e}")
        return

    # --- Listings ---
    if "price" in listings.columns:
        listings["price"] = (
            listings["price"]
            .astype(str)
            .str.replace(r"[\$,]", "", regex=True)
            .astype(float)
        )
        before = len(listings)
        listings = listings[listings["price"].between(PRICE_MIN, PRICE_MAX)]
        print(f"   Listings: {before - len(listings)} outliers prix supprimés")

    if {"latitude","longitude"} <= set(listings.columns):
        listings = listings.dropna(subset=["latitude","longitude"])

    # --- Calendar ---
    if "price" in calendar.columns:
        calendar["price"] = (
            calendar["price"]
            .astype(str)
            .str.replace(r"[\$,]", "", regex=True)
            .astype(float)
        )
        before = len(calendar)
        calendar = calendar[calendar["price"].between(PRICE_MIN, PRICE_MAX)]
        print(f"   Calendar: {before - len(calendar)} outliers prix supprimés")

    if "date" in calendar.columns:
        calendar["date"] = pd.to_datetime(calendar["date"], errors="coerce")

    if "available" in calendar.columns:
        calendar["available"] = calendar["available"].astype(str).str.lower().map(
            {"t":1, "f":0, "true":1, "false":0}
        )

    # --- Reviews ---
    if "date" in reviews.columns:
        reviews["date"] = pd.to_datetime(reviews["date"], errors="coerce")

    # --- Uniformisation ---
    for df in [listings, calendar, reviews]:
        df["city"] = city

    # --- Sauvegarde ---
    listings.to_csv(os.path.join(out_path,"listings_final.csv"), index=False)
    calendar.to_csv(os.path.join(out_path,"calendar_final.csv"), index=False)
    reviews.to_csv(os.path.join(out_path,"reviews_final.csv"), index=False)

    print(f"✅ {city} nettoyée → {out_path}")

if __name__ == "__main__":
    for city in ["paris","seattle"]:
        clean_city(city)
