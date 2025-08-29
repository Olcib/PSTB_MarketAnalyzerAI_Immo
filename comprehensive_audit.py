import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
OUTPUT_DIR = "results/comprehensive_audit/"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(f"{OUTPUT_DIR}/plots/").mkdir(parents=True, exist_ok=True)

def load_all_datasets():
    """
    Charge tous les datasets pour audit complet
    """
    print("AUDIT EXHAUSTIF DES FUITES DE DONNEES")
    print("="*50)
    
    datasets = {}
    
    # Paris enrichi
    try:
        df_paris = pd.read_csv("data/processed/enriched_paris_fixed.csv")
        datasets['paris_enriched'] = df_paris
        print(f"Paris enrichi: {df_paris.shape[0]:,} × {df_paris.shape[1]}")
    except FileNotFoundError:
        print("Fichier Paris enrichi introuvable")
    
    # Seattle enrichi  
    try:
        df_seattle = pd.read_csv("data/processed/enriched_seattle_fixed.csv")
        datasets['seattle_enriched'] = df_seattle
        print(f"Seattle enrichi: {df_seattle.shape[0]:,} × {df_seattle.shape[1]}")
    except FileNotFoundError:
        print("Fichier Seattle enrichi introuvable")
    
    return datasets

def identify_price_related_columns(df, dataset_name):
    """
    Identifie toutes les colonnes liées au prix
    """
    print(f"\nANALYSE DES COLONNES LIEES AU PRIX - {dataset_name.upper()}")
    print("="*60)
    
    # Patterns suspects
    price_patterns = [
        r'.*price.*', r'.*revenue.*', r'.*cost.*', r'.*amount.*',
        r'.*dollar.*', r'.*euro.*', r'.*fee.*', r'.*deposit.*'
    ]
    
    derived_patterns = [
        r'.*_diff_\d+.*', r'.*_rolling_\d+.*', r'.*_lag_\d+.*',
        r'.*_mean.*', r'.*_std.*', r'.*_shift.*'
    ]
    
    # Variables cibles légitimes
    legitimate_targets = ['price', 'weekly_price', 'monthly_price']
    
    # Identifier les colonnes suspectes
    price_related = []
    derived_features = []
    
    for col in df.columns:
        col_lower = col.lower()
        
        # Colonnes directement liées au prix
        for pattern in price_patterns:
            if re.match(pattern, col_lower):
                if col not in legitimate_targets:
                    price_related.append(col)
                break
        
        # Features dérivées (potentiellement du prix)
        for pattern in derived_patterns:
            if re.match(pattern, col_lower):
                derived_features.append(col)
                break
    
    print(f"Colonnes liées au prix détectées: {len(price_related)}")
    for col in price_related[:10]:
        print(f"  - {col}")
    if len(price_related) > 10:
        print(f"  ... et {len(price_related) - 10} autres")
    
    print(f"\nFeatures dérivées détectées: {len(derived_features)}")
    for col in derived_features[:15]:
        print(f"  - {col}")
    if len(derived_features) > 15:
        print(f"  ... et {len(derived_features) - 15} autres")
    
    return price_related, derived_features

def analyze_correlations_with_price(df, dataset_name, suspicious_cols):
    """
    Analyse les corrélations avec le prix
    """
    print(f"\nCORRELATIONS AVEC LE PRIX - {dataset_name.upper()}")
    print("="*45)
    
    # Identifier la variable prix
    price_col = None
    for col in ['price', 'revenue']:
        if col in df.columns:
            price_col = col
            break
    
    if not price_col:
        print("Variable prix introuvable")
        return {}
    
    # Calculer corrélations pour colonnes suspectes
    correlations = {}
    high_correlations = []
    
    for col in suspicious_cols:
        if col in df.columns and col != price_col:
            try:
                # Nettoyer les données
                col_data = pd.to_numeric(df[col], errors='coerce')
                price_data = pd.to_numeric(df[price_col], errors='coerce')
                
                # Calculer corrélation
                valid_mask = col_data.notna() & price_data.notna()
                if valid_mask.sum() > 100:
                    corr = col_data[valid_mask].corr(price_data[valid_mask])
                    if not np.isnan(corr):
                        correlations[col] = abs(corr)
                        if abs(corr) > 0.8:
                            high_correlations.append((col, corr))
            except:
                continue
    
    # Trier par corrélation
    sorted_correlations = dict(sorted(correlations.items(), key=lambda x: x[1], reverse=True))
    
    print("Top 10 corrélations élevées:")
    for i, (col, corr) in enumerate(list(sorted_correlations.items())[:10], 1):
        status = "🚨" if corr > 0.95 else "⚠️" if corr > 0.8 else "📊"
        print(f"  {i:2d}. {col}: {corr:.3f} {status}")
    
    print(f"\nFUITES CRITIQUES détectées (corrélation > 0.8): {len(high_correlations)}")
    for col, corr in high_correlations[:5]:
        print(f"  🚨 {col}: {corr:.3f}")
    
    return sorted_correlations, high_correlations

def create_whitelist_features(df, dataset_name):
    """
    Crée une liste blanche des features légitimes
    """
    print(f"\nCREATION LISTE BLANCHE - {dataset_name.upper()}")
    print("="*40)
    
    # Features intrinsèques légitimes
    intrinsic_features = [
        # Caractéristiques physiques
        'accommodates', 'bedrooms', 'beds', 'bathrooms',
        
        # Géolocalisation
        'latitude', 'longitude', 'neighbourhood_cleansed', 'neighbourhood_group_cleansed',
        
        # Type de propriété
        'property_type', 'room_type', 'bed_type',
        
        # Politiques
        'minimum_nights', 'maximum_nights', 'availability_30', 'availability_60', 
        'availability_90', 'availability_365',
        
        # Historique reviews (pas dérivés)
        'number_of_reviews', 'reviews_per_month',
        'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
        'review_scores_checkin', 'review_scores_communication', 'review_scores_location',
        'review_scores_value',
        
        # Temporel de base
        'month', 'day_of_week', 'year',
        
        # Hôte (pas financier)
        'host_response_time', 'host_response_rate', 'host_acceptance_rate',
        'host_is_superhost', 'host_total_listings_count',
        
        # Commodités
        'nb_amenities', 'amenities'
    ]
    
    # Vérifier disponibilité
    available_features = []
    for feature in intrinsic_features:
        if feature in df.columns:
            available_features.append(feature)
    
    print(f"Features légitimes disponibles: {len(available_features)}")
    
    # Grouper par catégorie
    categories = {
        'Physiques': ['accommodates', 'bedrooms', 'beds', 'bathrooms'],
        'Géographiques': ['latitude', 'longitude', 'neighbourhood_cleansed'],
        'Type': ['property_type', 'room_type'],
        'Politiques': [f for f in available_features if 'availability' in f or 'nights' in f],
        'Reviews': [f for f in available_features if 'review_scores' in f or 'reviews' in f],
        'Temporel': ['month', 'day_of_week', 'year'],
        'Hôte': [f for f in available_features if 'host' in f],
        'Commodités': ['nb_amenities', 'amenities']
    }
    
    for category, features in categories.items():
        category_available = [f for f in features if f in available_features]
        if category_available:
            print(f"\n{category} ({len(category_available)}):")
            for f in category_available[:5]:
                print(f"  ✓ {f}")
            if len(category_available) > 5:
                print(f"  ... et {len(category_available) - 5} autres")
    
    return available_features

def generate_audit_report(datasets_analysis):
    """
    Génère le rapport d'audit complet
    """
    print(f"\nGENERATION RAPPORT D'AUDIT")
    print("="*35)
    
    report_lines = []
    report_lines.append("# RAPPORT AUDIT EXHAUSTIF DES FUITES DE DONNEES")
    report_lines.append("=" * 60)
    report_lines.append("")
    
    total_critical_leaks = 0
    total_suspicious_features = 0
    
    for dataset_name, analysis in datasets_analysis.items():
        report_lines.append(f"## {dataset_name.upper()}")
        report_lines.append("")
        
        # Résumé des fuites
        critical_leaks = len(analysis.get('high_correlations', []))
        suspicious_features = len(analysis.get('price_related', [])) + len(analysis.get('derived_features', []))
        
        total_critical_leaks += critical_leaks
        total_suspicious_features += suspicious_features
        
        report_lines.append(f"- **Fuites critiques**: {critical_leaks}")
        report_lines.append(f"- **Features suspectes**: {suspicious_features}")
        report_lines.append(f"- **Features légitimes**: {len(analysis.get('whitelist', []))}")
        report_lines.append("")
        
        # Détail des fuites critiques
        if critical_leaks > 0:
            report_lines.append("### Fuites Critiques Détectées")
            for col, corr in analysis.get('high_correlations', [])[:10]:
                report_lines.append(f"- `{col}`: corrélation {corr:.3f}")
            report_lines.append("")
    
    # Résumé global
    report_lines.append("## RESUME GLOBAL")
    report_lines.append("")
    report_lines.append(f"- **Total fuites critiques**: {total_critical_leaks}")
    report_lines.append(f"- **Total features suspectes**: {total_suspicious_features}")
    report_lines.append("")
    
    # Recommandations
    report_lines.append("## RECOMMANDATIONS")
    report_lines.append("")
    if total_critical_leaks > 5:
        report_lines.append("🚨 **RECONSTRUCTION COMPLETE REQUISE**")
        report_lines.append("")
        report_lines.append("Les datasets sont massivement contaminés par des fuites de données.")
        report_lines.append("Tous les modèles précédents sont invalides et doivent être refaits.")
    else:
        report_lines.append("⚠️ **NETTOYAGE CIBLE REQUIS**")
        report_lines.append("")
        report_lines.append("Quelques fuites détectées, nettoyage possible.")
    
    report_lines.append("")
    report_lines.append("### Actions Immédiates")
    report_lines.append("1. Utiliser uniquement les features de la liste blanche")
    report_lines.append("2. Exclure toutes les variables _diff_, _rolling_, _lag_")
    report_lines.append("3. Reconstruire les modèles depuis zéro")
    report_lines.append("4. Valider l'absence de corrélations > 0.7 avec le prix")
    
    # Sauvegarder
    with open(f"{OUTPUT_DIR}/audit_exhaustif_report.md", 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Rapport sauvegardé: {OUTPUT_DIR}/audit_exhaustif_report.md")
    
    return total_critical_leaks, total_suspicious_features

def main():
    """
    Audit exhaustif principal
    """
    print("🔍 DEBUT AUDIT EXHAUSTIF - ACTION 1")
    print("=" * 50)
    
    start_time = pd.Timestamp.now()
    
    try:
        # Charger tous les datasets
        datasets = load_all_datasets()
        
        if not datasets:
            print("Aucun dataset trouvé pour l'audit")
            return
        
        # Analyser chaque dataset
        datasets_analysis = {}
        
        for dataset_name, df in datasets.items():
            print(f"\n{'='*60}")
            print(f"ANALYSE DATASET: {dataset_name.upper()}")
            print(f"{'='*60}")
            
            # Identifier colonnes suspectes
            price_related, derived_features = identify_price_related_columns(df, dataset_name)
            
            # Analyser corrélations
            all_suspicious = price_related + derived_features
            correlations, high_correlations = analyze_correlations_with_price(df, dataset_name, all_suspicious)
            
            # Créer liste blanche
            whitelist = create_whitelist_features(df, dataset_name)
            
            # Stocker résultats
            datasets_analysis[dataset_name] = {
                'price_related': price_related,
                'derived_features': derived_features,
                'correlations': correlations,
                'high_correlations': high_correlations,
                'whitelist': whitelist
            }
        
        # Générer rapport
        total_leaks, total_suspicious = generate_audit_report(datasets_analysis)
        
        # Résumé final
        duration = pd.Timestamp.now() - start_time
        
        print(f"\n{'='*60}")
        print("AUDIT EXHAUSTIF TERMINE")
        print(f"{'='*60}")
        print(f"Durée: {duration.total_seconds():.1f} secondes")
        print(f"Fuites critiques détectées: {total_leaks}")
        print(f"Features suspectes totales: {total_suspicious}")
        
        if total_leaks > 10:
            print("🚨 CONTAMINATION MASSIVE - Reconstruction complète requise")
        elif total_leaks > 0:
            print("⚠️ CONTAMINATION PARTIELLE - Nettoyage ciblé requis")
        else:
            print("✅ DONNEES PROPRES - Peut procéder à la modélisation")
        
        print(f"\n📁 Rapport complet: {OUTPUT_DIR}/audit_exhaustif_report.md")
        
        return datasets_analysis
        
    except Exception as e:
        print(f"\n❌ Erreur durant l'audit: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = main()