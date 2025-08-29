import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Configuration des graphiques
plt.style.use('default')
sns.set_palette("husl")

# ==========================================
# CONFIGURATION - FICHIERS DE RÉFÉRENCE
# ==========================================

# Fichiers principaux après corrections
PARIS_FILE = "data/processed/enriched_paris_fixed.csv"
SEATTLE_FILE = "data/processed/enriched_seattle_fixed.csv"

# Dossier de sortie pour les résultats
OUTPUT_DIR = "results/ml_baseline/"
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/plots/", exist_ok=True)

def explore_data_structure():
    """
    Explore la structure des données pour identifier les bonnes colonnes
    """
    print("🔍 EXPLORATION DE LA STRUCTURE DES DONNÉES")
    print("="*70)
    
    # Chargement des fichiers
    df_paris = pd.read_csv(PARIS_FILE)
    df_seattle = pd.read_csv(SEATTLE_FILE)
    
    print(f"✅ Paris : {df_paris.shape[0]:,} lignes × {df_paris.shape[1]} colonnes")
    print(f"✅ Seattle : {df_seattle.shape[0]:,} lignes × {df_seattle.shape[1]} colonnes")
    
    # Recherche des colonnes pertinentes
    print("\n🔍 Recherche des features ML dans Paris...")
    paris_cols = df_paris.columns.tolist()
    
    # Recherche par mots-clés
    occupancy_cols = [col for col in paris_cols if 'occup' in col.lower()]
    price_cols = [col for col in paris_cols if 'price' in col.lower()]
    rating_cols = [col for col in paris_cols if 'rating' in col.lower() or 'score' in col.lower()]
    amenity_cols = [col for col in paris_cols if 'amenity' in col.lower() or 'amenities' in col.lower()]
    temporal_cols = [col for col in paris_cols if col in ['month', 'day_of_week', 'year', 'day']]
    
    print(f"  📊 Occupancy-related: {occupancy_cols[:3]}...")
    print(f"  💰 Price-related: {price_cols[:3]}...")
    print(f"  ⭐ Rating-related: {rating_cols[:3]}...")
    print(f"  🏠 Amenity-related: {amenity_cols[:3]}...")
    print(f"  📅 Temporal: {temporal_cols}")
    
    print("\n🔍 Recherche des features ML dans Seattle...")
    seattle_cols = df_seattle.columns.tolist()
    
    seattle_occupancy = [col for col in seattle_cols if 'occup' in col.lower()]
    seattle_price = [col for col in seattle_cols if 'price' in col.lower()]
    seattle_rating = [col for col in seattle_cols if 'rating' in col.lower() or 'score' in col.lower()]
    seattle_amenity = [col for col in seattle_cols if 'amenity' in col.lower() or 'amenities' in col.lower()]
    seattle_temporal = [col for col in seattle_cols if col in ['month', 'day_of_week', 'year', 'day']]
    
    print(f"  📊 Occupancy-related: {seattle_occupancy[:3]}...")
    print(f"  💰 Price-related: {seattle_price[:3]}...")
    print(f"  ⭐ Rating-related: {seattle_rating[:3]}...")
    print(f"  🏠 Amenity-related: {seattle_amenity[:3]}...")
    print(f"  📅 Temporal: {seattle_temporal}")
    
    # Retourner les datasets et les colonnes identifiées
    feature_mapping = {
        'paris': {
            'occupancy': occupancy_cols[0] if occupancy_cols else None,
            'price': price_cols[0] if price_cols else None,
            'rating': rating_cols[0] if rating_cols else None,
            'amenities': amenity_cols[0] if amenity_cols else None,
            'month': 'month' if 'month' in paris_cols else None,
            'day_of_week': 'day_of_week' if 'day_of_week' in paris_cols else None
        },
        'seattle': {
            'occupancy': seattle_occupancy[0] if seattle_occupancy else None,
            'price': seattle_price[0] if seattle_price else None,
            'rating': seattle_rating[0] if seattle_rating else None,
            'amenities': seattle_amenity[0] if seattle_amenity else None,
            'month': 'month' if 'month' in seattle_cols else None,
            'day_of_week': 'day_of_week' if 'day_of_week' in seattle_cols else None
        }
    }
    
    return df_paris, df_seattle, feature_mapping

def create_ml_features(df_paris, df_seattle, feature_mapping):
    """
    Crée les features ML standardisées
    """
    print("\n🎯 CRÉATION DES FEATURES ML STANDARDISÉES")
    print("="*70)
    
    # Fonction pour créer les features d'une ville
    def prepare_city_data(df, city, mapping):
        print(f"\n🏙️  Traitement {city}...")
        
        df_ml = df.copy()
        df_ml['city'] = city
        
        # Mapping des colonnes vers des noms standardisés
        column_map = {}
        
        if mapping['occupancy']:
            column_map[mapping['occupancy']] = 'occupancy_rate'
            print(f"  📊 Occupancy: {mapping['occupancy']} → occupancy_rate")
        
        if mapping['price']:
            column_map[mapping['price']] = 'revenue'
            print(f"  💰 Price: {mapping['price']} → revenue")
        
        if mapping['rating']:
            column_map[mapping['rating']] = 'sentiment_score'
            print(f"  ⭐ Rating: {mapping['rating']} → sentiment_score")
        
        if mapping['amenities']:
            column_map[mapping['amenities']] = 'nb_amenities'
            print(f"  🏠 Amenities: {mapping['amenities']} → nb_amenities")
        
        # Renommer les colonnes
        df_ml = df_ml.rename(columns=column_map)
        
        # Vérifier les colonnes temporelles
        temporal_cols = ['month', 'day_of_week']
        for col in temporal_cols:
            if col in df_ml.columns:
                print(f"  📅 Temporal: {col} ✅")
            else:
                print(f"  📅 Temporal: {col} ❌")
        
        # Sélectionner les colonnes finales disponibles
        target_cols = ['occupancy_rate', 'revenue', 'sentiment_score', 'nb_amenities', 'month', 'day_of_week', 'city']
        available_cols = [col for col in target_cols if col in df_ml.columns]
        
        df_final = df_ml[available_cols].copy()
        
        print(f"  ✅ Features disponibles: {len(available_cols)-1}/6")
        print(f"  📊 Dimensions finales: {df_final.shape}")
        
        return df_final
    
    # Traitement des deux villes
    df_paris_ml = prepare_city_data(df_paris, 'Paris', feature_mapping['paris'])
    df_seattle_ml = prepare_city_data(df_seattle, 'Seattle', feature_mapping['seattle'])
    
    # Combinaison des datasets
    df_combined = pd.concat([df_paris_ml, df_seattle_ml], ignore_index=True)
    
    print(f"\n✅ Dataset combiné: {df_combined.shape[0]:,} lignes × {df_combined.shape[1]} colonnes")
    
    # Affichage du résumé des features
    print(f"\n📋 RÉSUMÉ DES FEATURES:")
    for col in df_combined.columns:
        if col != 'city':
            missing_pct = df_combined[col].isnull().sum() / len(df_combined) * 100
            unique_vals = df_combined[col].nunique()
            print(f"  • {col}: {unique_vals} valeurs uniques, {missing_pct:.1f}% manquantes")
    
    return df_combined

def prepare_ml_targets(df_combined):
    """
    Prépare les variables cibles
    """
    print(f"\n🎯 PRÉPARATION DES VARIABLES CIBLES")
    print("="*70)
    
    # Variable cible régression : revenue (price)
    if 'revenue' in df_combined.columns:
        y_regression = df_combined['revenue'].copy()
        print(f"✅ Régression target: revenue")
        print(f"  • Min: ${y_regression.min():.2f}")
        print(f"  • Max: ${y_regression.max():.2f}")
        print(f"  • Moyenne: ${y_regression.mean():.2f}")
        print(f"  • Valeurs manquantes: {y_regression.isnull().sum()}")
    else:
        print("❌ Pas de variable revenue trouvée")
        y_regression = None
    
    # Variable cible classification : "bien noté" vs "mal noté"
    if 'sentiment_score' in df_combined.columns:
        sentiment_col = df_combined['sentiment_score']
        
        # Déterminer le seuil automatiquement selon l'échelle
        max_score = sentiment_col.max()
        
        if max_score <= 5:  # Échelle 1-5
            threshold = 4.5
            y_classification = (sentiment_col >= threshold).astype(int)
            print(f"✅ Classification target: sentiment_score >= {threshold} (échelle 1-5)")
        elif max_score <= 100:  # Échelle 1-100
            threshold = 80
            y_classification = (sentiment_col >= threshold).astype(int)
            print(f"✅ Classification target: sentiment_score >= {threshold} (échelle 1-100)")
        else:
            threshold = sentiment_col.median()
            y_classification = (sentiment_col >= threshold).astype(int)
            print(f"✅ Classification target: sentiment_score >= {threshold:.2f} (médiane)")
        
        # Statistiques classification
        class_counts = y_classification.value_counts()
        print(f"  • Mal noté (0): {class_counts.get(0, 0):,} ({class_counts.get(0, 0)/len(y_classification)*100:.1f}%)")
        print(f"  • Bien noté (1): {class_counts.get(1, 0):,} ({class_counts.get(1, 0)/len(y_classification)*100:.1f}%)")
        
    else:
        print("❌ Pas de variable sentiment_score trouvée")
        y_classification = None
    
    return y_regression, y_classification

def preprocess_features(df_combined):
    """
    Preprocessing des features
    """
    print(f"\n🔧 PREPROCESSING DES FEATURES")
    print("="*70)
    
    # Préparer X en excluant les targets
    exclude_cols = ['revenue', 'sentiment_score']
    X_cols = [col for col in df_combined.columns if col not in exclude_cols]
    X = df_combined[X_cols].copy()
    
    print(f"📊 Features pour ML: {X.columns.tolist()}")
    
    # Encodage city si présent
    if 'city' in X.columns:
        le = LabelEncoder()
        X['city_encoded'] = le.fit_transform(X['city'].fillna('Unknown'))
        X = X.drop('city', axis=1)
        print(f"✅ City encodé: {X['city_encoded'].nunique()} valeurs")
    
    # Gestion valeurs manquantes
    print(f"\n📊 Valeurs manquantes:")
    missing_info = X.isnull().sum()
    for col, missing in missing_info.items():
        if missing > 0:
            print(f"  • {col}: {missing} ({missing/len(X)*100:.1f}%)")
    
    # Imputation
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    
    print(f"✅ Imputation terminée")
    
    # Normalisation
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_imputed),
        columns=X_imputed.columns,
        index=X_imputed.index
    )
    
    print(f"✅ Normalisation terminée")
    print(f"✅ Features finales: {X_scaled.columns.tolist()}")
    
    return X_scaled

def train_models(X, y_reg, y_clf):
    """
    Entraîne les modèles baseline
    """
    print(f"\n🤖 ENTRAÎNEMENT DES MODÈLES BASELINE")
    print("="*70)
    
    results = {}
    
    # RÉGRESSION
    if y_reg is not None and not y_reg.isnull().all():
        print(f"\n📊 RÉGRESSION (Prédiction Prix)")
        print("-" * 40)
        
        # Nettoyer les données pour la régression
        mask_reg = ~y_reg.isnull()
        X_reg = X[mask_reg]
        y_reg_clean = y_reg[mask_reg]
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_reg, y_reg_clean, test_size=0.2, random_state=42
        )
        
        print(f"  📊 Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")
        
        # Régression Linéaire
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)
        
        rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
        r2_lr = r2_score(y_test, y_pred_lr)
        cv_lr = cross_val_score(lr, X_train, y_train, cv=5, scoring='r2')
        
        print(f"  🔸 Régression Linéaire:")
        print(f"    • RMSE: ${rmse_lr:.2f}")
        print(f"    • R²: {r2_lr:.3f}")
        print(f"    • CV R²: {cv_lr.mean():.3f}±{cv_lr.std():.3f}")
        
        # Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        
        rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
        r2_rf = r2_score(y_test, y_pred_rf)
        cv_rf = cross_val_score(rf, X_train, y_train, cv=5, scoring='r2')
        
        print(f"  🌲 Random Forest:")
        print(f"    • RMSE: ${rmse_rf:.2f}")
        print(f"    • R²: {r2_rf:.3f}")
        print(f"    • CV R²: {cv_rf.mean():.3f}±{cv_rf.std():.3f}")
        
        results['regression'] = {
            'LinearRegression': {'rmse': rmse_lr, 'r2': r2_lr, 'cv_mean': cv_lr.mean(), 'cv_std': cv_lr.std()},
            'RandomForest': {'rmse': rmse_rf, 'r2': r2_rf, 'cv_mean': cv_rf.mean(), 'cv_std': cv_rf.std()},
            'test_data': (X_test, y_test, y_pred_lr, y_pred_rf)
        }
    
    # CLASSIFICATION
    if y_clf is not None and not y_clf.isnull().all():
        print(f"\n🎯 CLASSIFICATION (Bien noté vs Mal noté)")
        print("-" * 40)
        
        # Nettoyer les données pour la classification
        mask_clf = ~y_clf.isnull()
        X_clf = X[mask_clf]
        y_clf_clean = y_clf[mask_clf]
        
        if y_clf_clean.nunique() > 1:  # Vérifier qu'il y a au moins 2 classes
            # Split stratifié
            X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
                X_clf, y_clf_clean, test_size=0.2, random_state=42, stratify=y_clf_clean
            )
            
            # Random Forest Classifier
            rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_clf.fit(X_train_clf, y_train_clf)
            y_pred_clf = rf_clf.predict(X_test_clf)
            
            accuracy = rf_clf.score(X_test_clf, y_test_clf)
            cv_clf = cross_val_score(rf_clf, X_train_clf, y_train_clf, cv=5)
            
            print(f"  🌲 Random Forest Classifier:")
            print(f"    • Accuracy: {accuracy:.3f}")
            print(f"    • CV Accuracy: {cv_clf.mean():.3f}±{cv_clf.std():.3f}")
            
            # Matrice de confusion
            cm = confusion_matrix(y_test_clf, y_pred_clf)
            print(f"  📊 Matrice de confusion:")
            print(f"       Prédiction")
            print(f"    Réel  0   1")
            print(f"      0  {cm[0,0]:3d} {cm[0,1]:3d}")
            print(f"      1  {cm[1,0]:3d} {cm[1,1]:3d}")
            
            results['classification'] = {
                'accuracy': accuracy,
                'cv_mean': cv_clf.mean(),
                'cv_std': cv_clf.std(),
                'confusion_matrix': cm,
                'test_data': (X_test_clf, y_test_clf, y_pred_clf)
            }
        else:
            print("  ⚠️  Une seule classe détectée - classification impossible")
    
    return results

def create_plots(results):
    """
    Crée les graphiques d'évaluation
    """
    if not results:
        print("⚠️  Pas de résultats à visualiser")
        return
    
    print(f"\n📊 CRÉATION DES GRAPHIQUES")
    print("="*70)
    
    n_plots = len(results)
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
    
    if n_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # Graphique régression
    if 'regression' in results:
        ax = axes[plot_idx]
        X_test, y_test, y_pred_lr, y_pred_rf = results['regression']['test_data']
        
        ax.scatter(y_test, y_pred_lr, alpha=0.6, label=f"Linear Reg (R²={results['regression']['LinearRegression']['r2']:.3f})")
        ax.scatter(y_test, y_pred_rf, alpha=0.6, label=f"Random Forest (R²={results['regression']['RandomForest']['r2']:.3f})")
        
        min_val, max_val = y_test.min(), y_test.max()
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Parfait')
        
        ax.set_xlabel('Prix Réel ($)')
        ax.set_ylabel('Prix Prédit ($)')
        ax.set_title('Régression: Prédictions vs Réel')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plot_idx += 1
    
    # Graphique classification
    if 'classification' in results:
        ax = axes[plot_idx]
        cm = results['classification']['confusion_matrix']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Mal noté', 'Bien noté'],
                   yticklabels=['Mal noté', 'Bien noté'])
        ax.set_title(f"Classification (Acc: {results['classification']['accuracy']:.3f})")
        ax.set_xlabel('Prédiction')
        ax.set_ylabel('Réel')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plots/ml_baseline_results.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Graphiques sauvegardés: {OUTPUT_DIR}/plots/ml_baseline_results.png")

def generate_final_report(results):
    """
    Génère le rapport final
    """
    print(f"\n📋 GÉNÉRATION DU RAPPORT FINAL")
    print("="*70)
    
    report_lines = []
    report_lines.append("# 📊 RAPPORT ML BASELINE - JOUR 3")
    report_lines.append("=" * 50)
    report_lines.append("")
    
    # Résumé exécutif
    report_lines.append("## 🎯 RÉSUMÉ EXÉCUTIF")
    report_lines.append("")
    
    if 'regression' in results:
        best_reg_r2 = max(results['regression']['LinearRegression']['r2'], 
                         results['regression']['RandomForest']['r2'])
        best_reg_model = 'Random Forest' if results['regression']['RandomForest']['r2'] > results['regression']['LinearRegression']['r2'] else 'Linear Regression'
        
        report_lines.append("### 📊 Régression (Prédiction Prix)")
        report_lines.append("| Modèle | RMSE ($) | R² | CV R² |")
        report_lines.append("|--------|----------|----|----|")
        report_lines.append(f"| Linear Regression | ${results['regression']['LinearRegression']['rmse']:.2f} | {results['regression']['LinearRegression']['r2']:.3f} | {results['regression']['LinearRegression']['cv_mean']:.3f}±{results['regression']['LinearRegression']['cv_std']:.3f} |")
        report_lines.append(f"| Random Forest | ${results['regression']['RandomForest']['rmse']:.2f} | {results['regression']['RandomForest']['r2']:.3f} | {results['regression']['RandomForest']['cv_mean']:.3f}±{results['regression']['RandomForest']['cv_std']:.3f} |")
        report_lines.append("")
        report_lines.append(f"**🏆 Meilleur modèle**: {best_reg_model} (R² = {best_reg_r2:.3f})")
        report_lines.append("")
    
    if 'classification' in results:
        report_lines.append("### 🎯 Classification (Bien noté vs Mal noté)")
        report_lines.append("| Métrique | Valeur |")
        report_lines.append("|----------|--------|")
        report_lines.append(f"| Accuracy | {results['classification']['accuracy']:.3f} |")
        report_lines.append(f"| CV Accuracy | {results['classification']['cv_mean']:.3f}±{results['classification']['cv_std']:.3f} |")
        report_lines.append("")
        
        cm = results['classification']['confusion_matrix']
        report_lines.append("**📊 Matrice de confusion:**")
        report_lines.append("```")
        report_lines.append("           Prédiction")
        report_lines.append("Réel    Mal noté  Bien noté")
        report_lines.append(f"Mal noté    {cm[0,0]:3d}      {cm[0,1]:3d}")
        report_lines.append(f"Bien noté   {cm[1,0]:3d}      {cm[1,1]:3d}")
        report_lines.append("```")
        report_lines.append("")
    
    # Conclusions
    report_lines.append("## 🎯 CONCLUSIONS ET PROCHAINES ÉTAPES")
    report_lines.append("")
    
    if 'regression' in results:
        best_r2 = max(results['regression']['LinearRegression']['r2'], 
                     results['regression']['RandomForest']['r2'])
        if best_r2 > 0.7:
            report_lines.append("✅ **Régression**: Performance excellente - Prêt pour optimisation avancée")
        elif best_r2 > 0.5:
            report_lines.append("🔶 **Régression**: Performance correcte - Feature engineering recommandé")
        else:
            report_lines.append("❌ **Régression**: Performance faible - Analyse approfondie nécessaire")
    
    if 'classification' in results:
        acc = results['classification']['accuracy']
        if acc > 0.85:
            report_lines.append("✅ **Classification**: Performance excellente - Prêt pour optimisation")
        elif acc > 0.75:
            report_lines.append("🔶 **Classification**: Performance correcte - Tuning recommandé")
        else:
            report_lines.append("❌ **Classification**: Performance faible - Revoir stratégie")
    
    # Sauvegarde
    with open(f"{OUTPUT_DIR}/ml_baseline_report.md", 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✅ Rapport sauvegardé: {OUTPUT_DIR}/ml_baseline_report.md")
    
    # Affichage résumé console
    print(f"\n🏆 RÉSULTATS FINAUX")
    print("="*50)
    
    if 'regression' in results:
        best_reg_r2 = max(results['regression']['LinearRegression']['r2'], 
                         results['regression']['RandomForest']['r2'])
        best_reg_model = 'Random Forest' if results['regression']['RandomForest']['r2'] > results['regression']['LinearRegression']['r2'] else 'Linear Regression'
        print(f"📊 RÉGRESSION: {best_reg_model} - R² = {best_reg_r2:.3f}")
    
    if 'classification' in results:
        print(f"🎯 CLASSIFICATION: Random Forest - Accuracy = {results['classification']['accuracy']:.3f}")
    
    print("="*50)

def main():
    """
    Fonction principale ML Baseline - Version corrigée
    """
    print("🚀 JOUR 3 - ML BASELINE (VERSION CORRIGÉE)")
    print("="*70)
    print("Objectif : Adapter aux données réelles et entraîner modèles baseline")
    print()
    
    try:
        # 1. Explorer la structure des données
        df_paris, df_seattle, feature_mapping = explore_data_structure()
        
        # 2. Créer les features ML
        df_combined = create_ml_features(df_paris, df_seattle, feature_mapping)
        
        # 3. Préparer les targets
        y_regression, y_classification = prepare_ml_targets(df_combined)
        
        # 4. Preprocessing
        X = preprocess_features(df_combined)
        
        # 5. Entraînement
        results = train_models(X, y_regression, y_classification)
        
        # 6. Visualisation
        create_plots(results)
        
        # 7. Rapport final
        generate_final_report(results)
        
        print(f"\n🎉 ML BASELINE TERMINÉ AVEC SUCCÈS !")
        print(f"📁 Résultats disponibles dans: {OUTPUT_DIR}")
        print(f"📊 Fichiers créés:")
        print(f"  • {OUTPUT_DIR}/ml_baseline_report.md")
        print(f"  • {OUTPUT_DIR}/plots/ml_baseline_results.png")
        
    except Exception as e:
        print(f"\n❌ Erreur durant le ML Baseline: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()