#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
validate_features.py
--------------------
Contrôle qualité des features créées pour Paris et Seattle.
Affiche le % de valeurs remplies et manquantes pour chaque colonne de features.
"""

import pandas as pd

# Fichiers générés par le feature engineering
paris_fp = "data/features/paris_features.csv"
seattle_fp = "data/features/seattle_features.csv"

# Charger les datasets complets
print("📂 Chargement des datasets...")
paris_full = pd.read_csv(paris_fp)
seattle_full = pd.read_csv(seattle_fp)

print(f"✅ Paris: {len(paris_full)} lignes, {len(paris_full.columns)} colonnes")
print(f"✅ Seattle: {len(seattle_full)} lignes, {len(seattle_full.columns)} colonnes")

# Colonnes de features créées
features = ["occupancy_rate", "revenue", "sentiment_reviews", "nb_amenities", "day_of_week", "month"]

def feature_stats(df, city):
    """Calcule les statistiques de complétude pour chaque feature."""
    stats = []
    total = len(df)
    
    for col in features:
        if col in df.columns:
            non_nulls = df[col].notna().sum()
            nulls = total - non_nulls
            pct_non_null = round(100 * non_nulls / total, 2)
            pct_null = round(100 * nulls / total, 2)
            
            stats.append({
                "feature": col,
                "city": city,
                "total": total,
                "non_null_count": non_nulls,
                "null_count": nulls,
                "% non-null": pct_non_null,
                "% null": pct_null
            })
        else:
            stats.append({
                "feature": col,
                "city": city,
                "total": total,
                "non_null_count": 0,
                "null_count": total,
                "% non-null": 0.0,
                "% null": 100.0
            })
    
    return pd.DataFrame(stats)

# Calcul pour Paris et Seattle
print("\n🔍 Calcul des statistiques...")
paris_stats = feature_stats(paris_full, "Paris")
seattle_stats = feature_stats(seattle_full, "Seattle")

# Fusion des résultats
all_stats = pd.concat([paris_stats, seattle_stats], ignore_index=True)

# Affichage des résultats
print("\n" + "="*80)
print("📊 COMPLÉTUDE DES FEATURES CRÉÉES (Paris vs Seattle)")
print("="*80)

# Résumé par feature
for feature in features:
    print(f"\n🔸 {feature.upper()}:")
    feature_data = all_stats[all_stats['feature'] == feature]
    
    for _, row in feature_data.iterrows():
        city = row['city']
        total = row['total']
        non_null = row['non_null_count']
        pct = row['% non-null']
        
        print(f"   {city:8}: {non_null:5}/{total:5} ({pct:5.1f}%)")

# Résumé par ville
print("\n" + "="*80)
print("📈 RÉSUMÉ PAR VILLE")
print("="*80)

for city in ["Paris", "Seattle"]:
    city_data = all_stats[all_stats['city'] == city]
    total_features = len(city_data)
    complete_features = len(city_data[city_data['% non-null'] > 0])
    
    print(f"\n🏙️ {city.upper()}:")
    print(f"   Total des listings: {city_data['total'].iloc[0]:,}")
    print(f"   Features disponibles: {complete_features}/{total_features}")
    
    # Détail des features
    for _, row in city_data.iterrows():
        feature = row['feature']
        pct = row['% non-null']
        status = "✅" if pct > 0 else "❌"
        print(f"   {status} {feature:20}: {pct:5.1f}%")

print("\n" + "="*80)
print("🎯 RECOMMANDATIONS")
print("="*80)

# Analyse des features manquantes
missing_features = all_stats[all_stats['% non-null'] == 0]
if not missing_features.empty:
    print("\n❌ Features manquantes détectées:")
    for _, row in missing_features.iterrows():
        print(f"   - {row['city']}: {row['feature']}")
    print("\n💡 Vérifiez les données sources et le processus de feature engineering.")
else:
    print("\n✅ Toutes les features sont disponibles pour les deux villes!")

print("\n" + "="*80)