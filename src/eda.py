#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eda.py
------
Exploratory Data Analysis (EDA) for Airbnb datasets (Paris, Seattle).
- Gère fallback RAW si processed indisponible
- Génère un rapport Excel complet (Résumé + describe détaillé par dataset)
- Crée une fiche de synthèse par ville
- Vérifie NA, doublons, prix, outliers
- Calcule taux d’occupation mensuel si calendar dispo
- Produit des graphiques (prix, reviews, occupancy)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

PROC_DIR = "data/processed"
RAW_DIR = "data/raw"
REPORT_FILE = os.path.join(PROC_DIR, "eda_report.xlsx")

# seuils
NA_THRESHOLD = 0.8
PRICE_MIN, PRICE_MAX = 10, 2000

report_rows = []   # résumé par ville
all_details = {}   # describe détaillé par dataset


def safe_path(base_path, fname_clean, fname_raw):
    """Retourne le chemin du fichier clean si dispo, sinon raw"""
    clean = os.path.join(base_path, fname_clean)
    raw = os.path.join(base_path, fname_raw)
    if os.path.exists(clean):
        return clean
    elif os.path.exists(raw):
        return raw
    else:
        return None


def analyze_city(city: str):
    print(f"\n📊 EDA for {city}")

    # --------------------------
    # 1. Chemins datasets
    # --------------------------
    base_proc = os.path.join(PROC_DIR, city)
    base_raw = os.path.join(RAW_DIR, city)

    base_path = base_proc if os.path.exists(base_proc) and os.listdir(base_proc) else base_raw
    if not os.path.exists(base_path):
        print(f"❌ Aucun dossier trouvé pour {city}")
        return None

    listings_fp = safe_path(base_path, "listings_clean.csv", "listings.csv")
    calendar_fp = safe_path(base_path, "calendar_clean.csv", "calendar.csv")
    reviews_fp = safe_path(base_path, "reviews_clean.csv", "reviews.csv")

    if not listings_fp or not calendar_fp or not reviews_fp:
        print(f"❌ Impossible de trouver tous les fichiers nécessaires pour {city}")
        return None

    # --------------------------
    # 2. Chargement
    # --------------------------
    listings = pd.read_csv(listings_fp, low_memory=False)
    calendar = pd.read_csv(calendar_fp, low_memory=False)
    reviews = pd.read_csv(reviews_fp, low_memory=False)

    # conversion prix si string
    if "price" in listings.columns and listings["price"].dtype == "object":
        listings["price"] = listings["price"].replace('[\$,]', '', regex=True).astype(float)

    # --------------------------
    # 3. Infos générales
    # --------------------------
    listings_shape = f"{listings.shape[0]}×{listings.shape[1]}"
    calendar_shape = f"{calendar.shape[0]}×{calendar.shape[1]}"
    reviews_shape = f"{reviews.shape[0]}×{reviews.shape[1]}"

    cal_range = f"{calendar['date'].min()} → {calendar['date'].max()}" if not calendar.empty and "date" in calendar.columns else "vide"
    rev_range = f"{reviews['date'].min()} → {reviews['date'].max()}" if not reviews.empty and "date" in reviews.columns else "vide"

    # --------------------------
    # 4. Valeurs manquantes
    # --------------------------
    na_pct = listings.isna().mean().sort_values(ascending=False)
    top_na = na_pct.head(5).to_dict()
    keep_cols = [c for c in listings.columns if listings[c].isna().mean() < NA_THRESHOLD]
    drop_cols = [c for c in listings.columns if listings[c].isna().mean() >= NA_THRESHOLD]

    # --------------------------
    # 5. Doublons
    # --------------------------
    dups = listings.duplicated("id").sum() if "id" in listings.columns else listings.duplicated().sum()
    dups_pct = dups / len(listings) * 100

    # --------------------------
    # 6. Prix & outliers
    # --------------------------
    price_stats, outliers_n = {}, 0
    if "price" in listings.columns:
        prices = listings["price"].dropna()
        prices = prices[prices < 5000]  # filtre soft pour stats
        if not prices.empty:
            desc = prices.describe(percentiles=[0.25, 0.5, 0.75]).to_dict()
            price_stats = {
                "min": desc["min"],
                "q1": desc["25%"],
                "median": desc["50%"],
                "mean": desc["mean"],
                "q3": desc["75%"],
                "max": desc["max"],
                "std": desc["std"],
            }
        # outliers stricts
        outliers_n = len(listings[(listings["price"] < PRICE_MIN) | (listings["price"] > PRICE_MAX)])

    # --------------------------
    # 7. Occupancy mensuel
    # --------------------------
    if not calendar.empty and "available" in calendar.columns:
        calendar["date"] = pd.to_datetime(calendar["date"], errors="coerce")
        calendar["available_flag"] = (calendar["available"] == "f").astype(int)
        occupancy = calendar.groupby(calendar["date"].dt.to_period("M"))["available_flag"].mean()
        if not occupancy.empty:
            occupancy.plot(kind="bar", title=f"Taux d’occupation mensuel - {city}")
            plt.tight_layout()
            plt.savefig(os.path.join(base_path, "occupancy_trend.png"))
            plt.close()

    # --------------------------
    # 8. Describe détaillé (sans datetime_is_numeric)
    # --------------------------
    try:
        listings_info = listings.describe(include="all").transpose()
    except Exception:
        listings_info = listings.describe().transpose()

    try:
        calendar_info = calendar.describe(include="all").transpose()
    except Exception:
        calendar_info = calendar.describe().transpose()

    try:
        reviews_info = reviews.describe(include="all").transpose()
    except Exception:
        reviews_info = reviews.describe().transpose()

    # Sauvegarde
    all_details[f"{city}_listings"] = listings_info
    all_details[f"{city}_calendar"] = calendar_info
    all_details[f"{city}_reviews"] = reviews_info

    # --------------------------
    # 9. Recommandations
    # --------------------------
    reco = []
    if calendar.empty:
        reco.append("⚠️ Calendar vide → recharger snapshot")
    if na_pct.max() > 0.5:
        reco.append("⚠️ Colonnes avec >50% NA → supprimer ou ignorer")
    if outliers_n > 0:
        reco.append(f"⚠️ {outliers_n} outliers prix (<{PRICE_MIN}€ ou >{PRICE_MAX}€) → filtrer")
    if not reco:
        reco.append("✅ Données globalement robustes")

    # --------------------------
    # 10. Ajout résumé global
    # --------------------------
    report_rows.append({
        "Ville": city,
        "Listings": listings_shape,
        "Calendar": f"{calendar_shape} | {cal_range}",
        "Reviews": f"{reviews_shape} | {rev_range}",
        "Top 5 NA (%)": top_na,
        "Doublons": f"{dups} ({dups_pct:.1f}%)",
        "Prix/nuit": price_stats,
        "Colonnes gardées": len(keep_cols),
        "Colonnes supprimées": drop_cols,
        "Outliers prix": outliers_n,
        "Recommandations": "; ".join(reco),
    })

    # --------------------------
    # 11. Graphiques simples
    # --------------------------
    if "price" in listings.columns:
        listings["price"].hist(bins=50)
        plt.title(f"Distribution des prix - {city}")
        plt.xlabel("Prix/nuit")
        plt.ylabel("Nb annonces")
        plt.savefig(os.path.join(base_path, "price_distribution.png"))
        plt.close()

    if not reviews.empty and "comments" in reviews.columns:
        reviews["review_length"] = reviews["comments"].astype(str).apply(len)
        reviews["review_length"].hist(bins=50)
        plt.title(f"Distribution longueur reviews - {city}")
        plt.savefig(os.path.join(base_path, "reviews_length.png"))
        plt.close()

    # --------------------------
    # 12. Export fiche EDA par ville
    # --------------------------
    synthesis = [
        {"Étape": "Exploration initiale", "Résultats": f"{listings.shape[0]} lignes, {listings.shape[1]} colonnes"},
        {"Étape": "Nettoyage", "Résultats": f"{dups} doublons ; top NA: {list(top_na.keys())}"},
        {"Étape": "Analyse prix", "Résultats": f"Moy: {price_stats.get('mean',0):.2f}, Méd: {price_stats.get('median',0):.2f}"},
        {"Étape": "Recommandations", "Résultats": '; '.join(reco)},
    ]
    pd.DataFrame(synthesis).to_excel(f"EDA_synthesis_{city}.xlsx", index=False)

    return True


if __name__ == "__main__":
    for city in ["paris", "seattle"]:
        analyze_city(city)

    # --------------------------
    # 13. Export comparatif Excel global
    # --------------------------
    with pd.ExcelWriter(REPORT_FILE) as writer:
        pd.DataFrame(report_rows).to_excel(writer, sheet_name="Résumé", index=False)
        for name, df in all_details.items():
            df.to_excel(writer, sheet_name=name[:31])

    print(f"\n✅ Rapport EDA exporté → {REPORT_FILE}")
