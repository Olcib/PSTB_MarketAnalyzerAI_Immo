#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check Features Columns
----------------------
Compare les colonnes présentes dans les fichiers générés
avec celles attendues par validate_features.py
"""

import pandas as pd
import os

FEATURE_DIR = "data/features"
EXPECTED = ["occupancy_rate", "revenue", "sentiment_reviews",
            "nb_amenities", "day_of_week", "month"]

for city in ["paris", "seattle"]:
    fp = os.path.join(FEATURE_DIR, f"{city}_features.csv")
    if not os.path.exists(fp):
        print(f"❌ Fichier manquant: {fp}")
        continue

    df = pd.read_csv(fp)
    cols = df.columns.tolist()
    print(f"\n=== {city.upper()} ({len(df)} lignes, {len(cols)} colonnes) ===")
    print("Colonnes présentes:", cols[:15], "...")  # afficher seulement les 15 premières
    missing = [c for c in EXPECTED if c not in df.columns]
    if missing:
        print("❌ Colonnes manquantes:", missing)
    else:
        print("✅ Toutes les colonnes attendues sont présentes")
