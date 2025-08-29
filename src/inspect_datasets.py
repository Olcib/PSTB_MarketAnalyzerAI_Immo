#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspect_datasets.py
-------------------
Affiche les 10 premières et 10 dernières lignes
des datasets nettoyés (listings et reviews) pour Paris et Seattle.
"""

import os
import pandas as pd

FILES_TO_INSPECT = {
    "paris": [
        "listings_clean8000.csv",   # <- Vérifie bien le nom exact (s à listings ?)
        "reviews_clean8000.csv"
    ],
    "seattle": [
        "listings_clean.csv",
        "reviews_clean20000.csv"
    ]
}

def inspect_file(city: str, filename: str):
    """
    Affiche head(10) et tail(10) d'un fichier CSV s'il existe.
    Corrige les problèmes de parsing pour les reviews.
    """
    fp = os.path.join("data", "processed", city, filename)

    if not os.path.exists(fp):
        print(f"❌ Fichier manquant : {fp}")
        return

    print(f"\n📂 {city.upper()} — {filename}")
    try:
        df = pd.read_csv(fp, sep=",", quotechar='"', on_bad_lines="skip", low_memory=False)
    except Exception as e:
        print(f"⚠️ Erreur de lecture pour {filename} : {e}")
        return

    print(f"Dimensions: {df.shape[0]} lignes × {df.shape[1]} colonnes")

    print("\n🔹 Top 10 lignes (head):")
    print(df.head(10))

    print("\n🔹 Bottom 10 lignes (tail):")
    print(df.tail(10))
    print("-" * 80)


if __name__ == "__main__":
    for city, files in FILES_TO_INSPECT.items():
        for fname in files:
            inspect_file(city, fname)
