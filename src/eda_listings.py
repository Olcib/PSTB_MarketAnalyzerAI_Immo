#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eda_listings.py
---------------
Exploratory Data Analysis pour listings.csv
"""

import pandas as pd

def eda_listings(city, fp):
    print(f"\n📊 EDA LISTINGS — {city.upper()} ({fp})")
    df = pd.read_csv(fp)

    # Dimensions et colonnes
    print(f"Dimensions: {df.shape[0]} lignes × {df.shape[1]} colonnes")
    print("Colonnes:", list(df.columns))

    # Types des colonnes
    print("\n🔹 Info (types des colonnes):")
    print(df.info())

    # Valeurs manquantes
    print("\n🔹 Valeurs manquantes (top 10):")
    print(df.isnull().mean().sort_values(ascending=False).head(10))

    # Doublons
    print("\n🔹 Doublons:", df.duplicated().sum())

    # Statistiques descriptives
    print("\n🔹 Statistiques descriptives:")
    print(df.describe(include="all").transpose().head(15))  # affiche les 15 premières lignes

    # Prix (si colonne existe)
    if "price" in df.columns:
        try:
            prices = df["price"].replace("[\$,]", "", regex=True).astype(float)
            prices = prices[prices < 5000]  # supprime outliers
            print("\n🔹 Distribution prix (<5000):")
            print(prices.describe(percentiles=[.25, .5, .75, .9]))
        except Exception as e:
            print(f"⚠️ Impossible d’analyser les prix: {e}")
