#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Engineering
-------------------
Génère les features dérivées pour Paris & Seattle :
- occupancy_rate
- revenue
- sentiment_reviews
- nb_amenities
- effets calendaires (day_of_week, month)
"""

import os
import pandas as pd
from textblob import TextBlob

CLEAN_DIR = "data/clean"
FEAT_DIR = "data/features"
os.makedirs(FEAT_DIR, exist_ok=True)


def compute_sentiment(text):
    """Calcule le score de sentiment d’un commentaire."""
    try:
        return TextBlob(str(text)).sentiment.polarity
    except Exception:
        return 0


def build_features(city: str):
    print(f"\n🧩 Feature engineering → {city}")
    path = os.path.join(CLEAN_DIR, city)

    # fichiers sources
    listings_fp = os.path.join(path, "listings_final.csv")
    calendar_fp = os.path.join(path, "calendar_final.csv")
    reviews_fp  = os.path.join(path, "reviews_final.csv")


    # chargement datasets
    listings = pd.read_csv(listings_fp, low_memory=False)
    calendar = pd.read_csv(calendar_fp, low_memory=False)
    reviews = pd.read_csv(reviews_fp, low_memory=False)

    # ---- 1) occupancy_rate ----
    try:
        occ = (calendar.assign(avail_flag=calendar["available"].map(lambda x: 1 if str(x).lower() in ["t", "true", "1", "yes"] else 0))
                      .groupby("listing_id")["avail_flag"].mean()
                      .rename("occupancy_rate"))
        listings = listings.merge(occ, left_on="id", right_index=True, how="left")
    except Exception as e:
        print(f"⚠️ occupancy_rate error: {e}")

    # ---- 2) revenue ----
    try:
        cal = calendar.copy()
        cal["price"] = (cal["price"].astype(str)
                                      .str.replace("[$,]", "", regex=True)
                                      .astype(float))
        rev = (cal.assign(revenue=cal["price"] * cal["available"].map(lambda x: 1 if str(x).lower() in ["t","true","1","yes"] else 0))
                   .groupby("listing_id")["revenue"].sum()
                   .rename("revenue"))
        listings = listings.merge(rev, left_on="id", right_index=True, how="left")
    except Exception as e:
        print(f"⚠️ revenue error: {e}")

    # ---- 3) sentiment_reviews ----
    try:
        sent = (reviews.groupby("listing_id")["comments"]
                       .apply(lambda x: x.dropna().astype(str).apply(compute_sentiment).mean())
                       .rename("sentiment_reviews"))
        listings = listings.merge(sent, left_on="id", right_index=True, how="left")
    except Exception as e:
        print(f"⚠️ sentiment_reviews error: {e}")

    # ---- 4) nb_amenities ----
    if "amenities" in listings.columns:
        listings["nb_amenities"] = listings["amenities"].astype(str).apply(lambda x: len(x.split(",")))
    else:
        listings["nb_amenities"] = 0

    # ---- 5) effets calendaires ----
    try:
        cal = calendar.copy()
        cal["date"] = pd.to_datetime(cal["date"], errors="coerce")
        day_occ = (cal.groupby(cal["date"].dt.day_name())["available"]
                      .count()
                      .rename("day_of_week"))
        listings["day_of_week"] = cal["date"].dt.day_name().mode()[0] if not cal.empty else None
        listings["month"] = cal["date"].dt.month.mode()[0] if not cal.empty else None
    except Exception as e:
        print(f"⚠️ effets calendaires error: {e}")

    # Sauvegarde
    out_fp = os.path.join(FEAT_DIR, f"{city}_features.csv")
    listings.to_csv(out_fp, index=False)
    print(f"✅ Fichier features sauvegardé: {out_fp} ({listings.shape[0]} lignes, {listings.shape[1]} colonnes)")


def main():
    for city in ["paris", "seattle"]:
        build_features(city)


if __name__ == "__main__":
    main()
