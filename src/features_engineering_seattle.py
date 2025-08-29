#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
features_engineering_seattle.py
-------------------------------------
Construction des features spécifiques pour Seattle.
Entrées : data/processed/seattle/listings.csv, calendar.csv, reviews.csv
Sortie : data/features/seattle_features.csv (+ seattle_month_occ.csv)
"""

import pandas as pd
from pathlib import Path
from textblob import TextBlob


# =====================================================
# Helper functions
# =====================================================
def compute_sentiment(text: str) -> float:
    """Retourne un score de sentiment (-1 à 1) basé sur TextBlob."""
    try:
        return TextBlob(str(text)).sentiment.polarity
    except Exception:
        return 0.0


# =====================================================
# Feature engineering Seattle
# =====================================================
def build_features(processed_dir="data/processed/seattle", features_dir="data/features"):
    proc_path = Path(processed_dir)
    feat_path = Path(features_dir)
    feat_path.mkdir(parents=True, exist_ok=True)

    # Charger les fichiers CSV
    listings_fp = proc_path / "listings.csv"
    calendar_fp = proc_path / "calendar.csv"
    reviews_fp = proc_path / "reviews.csv"

    if not (listings_fp.exists() and calendar_fp.exists() and reviews_fp.exists()):
        raise FileNotFoundError(f"❌ Fichiers manquants dans {proc_path}")

    listings = pd.read_csv(listings_fp, low_memory=False, on_bad_lines="skip")
    calendar = pd.read_csv(calendar_fp, low_memory=False, on_bad_lines="skip")
    reviews = pd.read_csv(reviews_fp, low_memory=False, on_bad_lines="skip")

    # Nettoyage des dates calendrier
    if "date" in calendar.columns:
        calendar["date"] = pd.to_datetime(calendar["date"], errors="coerce")
        calendar = calendar.dropna(subset=["date"])

    # ================== Features listings ==================
    if "amenities" in listings.columns:
        listings["amenities_count"] = listings["amenities"].apply(
            lambda x: len(str(x).split(",")) if pd.notnull(x) else 0
        )
    else:
        listings["amenities_count"] = 0

    if "host_is_superhost" in listings.columns:
        listings["is_superhost"] = listings["host_is_superhost"].map({"t": 1, "f": 0}).fillna(0)
    else:
        listings["is_superhost"] = 0

    # Vérification clé primaire
    if "id" in listings.columns:
        id_col = "id"
    elif "listing_id" in listings.columns:
        id_col = "listing_id"
    else:
        listings.insert(0, "listing_id", range(1, len(listings) + 1))
        id_col = "listing_id"
        print("⚠️ Aucun identifiant trouvé → listing_id artificiel créé")

    # ================== Features calendar ==================
    occ, avg_price, revenue = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if not calendar.empty and "available" in calendar.columns:
        calendar["available"] = calendar["available"].map({"t": 1, "f": 0}).fillna(calendar["available"])
        calendar["available"] = pd.to_numeric(calendar["available"], errors="coerce").fillna(0)

        occ = calendar.groupby("listing_id")["available"].apply(lambda x: 1 - x.mean())
        occ.name = "occupancy_rate"

    if not calendar.empty and "price" in calendar.columns:
        calendar["price"] = (
            calendar["price"].astype(str).str.replace(r"[$,]", "", regex=True).astype(float)
        )
        avg_price = calendar.groupby("listing_id")["price"].mean().rename("avg_price")

    if not avg_price.empty and not occ.empty:
        revenue = pd.DataFrame({
            "est_revenue": avg_price * occ * 30  # estimation mensuelle
        })

    # ================== Features reviews ==================
    sentiment = pd.DataFrame()
    if not reviews.empty and "comments" in reviews.columns:
        reviews["sentiment_score"] = reviews["comments"].apply(compute_sentiment)
        sentiment = reviews.groupby("listing_id")["sentiment_score"].mean()

    # ================== Merge ==================
    feats = listings.copy()

    if not occ.empty:
        feats = feats.merge(occ, left_on=id_col, right_index=True, how="left")
    if not avg_price.empty:
        feats = feats.merge(avg_price, left_on=id_col, right_index=True, how="left")
    if not revenue.empty:
        feats = feats.merge(revenue, left_on=id_col, right_index=True, how="left")
    if not sentiment.empty:
        feats = feats.merge(sentiment, left_on=id_col, right_index=True, how="left")

    # Relative price
    if "avg_price" in feats.columns and feats["avg_price"].notna().any():
        median_price = feats["avg_price"].median()
        feats["price_relative"] = feats["avg_price"] / median_price

    # Monthly occupancy
    if not calendar.empty and "available" in calendar.columns:
        calendar["month"] = calendar["date"].dt.month
        month_occ = calendar.groupby(["listing_id", "month"])["available"].apply(lambda x: 1 - x.mean()).reset_index()
        month_occ.rename(columns={"available": "monthly_occ"}, inplace=True)
        month_occ.to_csv(feat_path / "seattle_month_occ.csv", index=False, encoding="utf-8")

    # Sauvegarde
    out_path = feat_path / "seattle_features.csv"
    feats.to_csv(out_path, index=False, encoding="utf-8")
    print(f"✅ Features Seattle sauvegardées dans {out_path}")


# =====================================================
# Main
# =====================================================
if __name__ == "__main__":
    try:
        build_features()
    except Exception as e:
        print(f"⚠️ Erreur Seattle: {e}")
