#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspect_calendar.py
-------------------
Affiche les 10 premières et 10 dernières lignes
des fichiers calendar_clean.csv pour Paris et Seattle.
"""

import os
import pandas as pd

def inspect_calendar(city: str):
    """
    Affiche head(10) et tail(10) pour le fichier calendar_clean.csv d'une ville
    """
    fp = os.path.join("data", "processed", city, "calendar_clean.csv")

    if not os.path.exists(fp):
        print(f"❌ Fichier manquant pour {city} : {fp}")
        return
    
    print(f"\n📂 {city.upper()} — calendar_clean.csv")
    df = pd.read_csv(fp)

    print("\n🔹 Top 10 lignes (head):")
    print(df.head(10))

    print("\n🔹 Bottom 10 lignes (tail):")
    print(df.tail(10))

if __name__ == "__main__":
    for city in ["paris", "seattle"]:
        inspect_calendar(city)
