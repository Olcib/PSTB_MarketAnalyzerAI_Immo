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

def explore_and_prepare_data():
    """
    Explore et prépare les données pour ML (version simplifiée et robuste)
    """
    print("🔍 EXPLORATION ET PRÉPARATION DES DONNÉES")
    print("="*70)
    
    # Chargement des fichiers
    df_paris = pd.read_csv(PARIS_FILE)
    df_seattle = pd.read_csv(SEATTLE_FILE)
    
    print(f"✅ Paris : {df_paris.shape[0]:,} lignes × {df_paris.shape[1]} colonnes")
    print(f"✅ Seattle : {df_seattle.shape[0]:,} lignes × {df_seattle.shape[1]} colonnes")
    
    # Préparation Paris
    print(f"\n🏙️  Préparation Paris...")
    paris_ml = df_paris.copy()
    paris_ml['city'] = 'Paris'
    
    # Mapping des colonnes Paris
    if 'estimated_occupancy_l365d' in paris_ml.columns:
        paris_ml['occupancy_rate'] = paris_ml['estimated_occupancy_l365d']
        print("  📊 Occupancy: estimated_occupancy_l365d → occupancy_rate")
    
    if 'price' in paris_ml.columns:
        paris_ml['revenue'] = paris_ml['price']
        print("  💰 Revenue: price → revenue")
    
    if 'review_scores_rating' in paris_ml.columns:
        paris_ml['sentiment_score'] = paris_ml['review_scores_rating']
        print("  ⭐ Sentiment: review_scores_rating → sentiment_score")
    
    # Utiliser nb_amenities directement
    if 'nb_amenities' in paris_ml.columns:
        print("  🏠 Amenities: nb_amenities ✅")
    elif 'amenities' in paris_ml.columns:
        paris_ml['nb_amenities'] = paris_ml['amenities']
        print("  🏠 Amenities: amenities → nb_amenities")
    
    # Temporel
    temporal_found = []
    for col in ['month', 'day_of_week']:
        if col in paris_ml.columns:
            temporal_found.append(col)
    print(f"  📅 Temporal: {temporal_found}")
    
    # Préparation Seattle
    print(f"\n🏙️  Préparation Seattle...")
    seattle_ml = df_seattle.copy()
    seattle_ml['city'] = 'Seattle'
    
    # Seattle a déjà occupancy_rate
    if 'occupancy_rate' in seattle_ml.columns:
        print("  📊 Occupancy: occupancy_rate ✅")
    
    if 'price' in seattle_ml.columns:
        seattle_ml['revenue'] = seattle_ml['price']
        print("  💰 Revenue: price → revenue")
    
    if 'review_scores_rating' in seattle_ml.columns:
        seattle_ml['sentiment_score'] = seattle_ml['review_scores_rating']
        print("  ⭐ Sentiment: review_scores_rating → sentiment_score")
    
    if 'nb_amenities' in seattle_ml.columns:
        print("  🏠 Amenities: nb_amenities ✅")
    elif 'amenities' in seattle_ml.columns:
        seattle_ml['nb_amenities'] = seattle_ml['amenities']
        print("  🏠 Amenities: amenities → nb_amenities")
    
    # Temporel Seattle
    seattle_temporal = []
    for col in ['month', 'day_of_week']:
        if col in seattle_ml.columns:
            seattle_temporal.append(col)
    print(f"  📅 Temporal: {seattle_temporal}")
    
    # Sélection des colonnes finales
    target_cols = ['occupancy_rate', 'revenue', 'sentiment_score', 'nb_amenities', 'month', 'day_of_week', 'city']
    
    # Paris
    paris_available = [col for col in target_cols if col in paris_ml.columns]
    df_paris_final = paris_ml[paris_available].copy()
    
    # Seattle  
    seattle_available = [col for col in target_cols if col in seattle_ml.columns]
    df_seattle_final = seattle_ml[seattle_available].copy()
    
    print(f"\n✅ Paris final: {len(paris_available)-1}/6 features ({df_paris_final.shape})")
    print(f"✅ Seattle final: {len(seattle_available)-1}/6 features ({df_seattle_final.shape})")
    
    # Combinaison
    df_combined = pd.concat([df_paris_final, df_seattle_final], ignore_index=True)
    print(f"✅ Dataset combiné: {df_combined.shape[0]:,} lignes × {df_combined.shape[1]} colonnes")
    
    # Résumé des features (version corrigée)
    print(f"\n📋 RÉSUMÉ DES FEATURES:")
    for col in df_combined.columns:
        if col != 'city':
            try:
                missing_count = df_combined[col].isnull().sum()
                total_count = len(df_combined)
                missing_pct = (missing_count / total_count) * 100
                unique_count = df_combined[col].nunique()
                print(f"  • {col}: {unique_count} valeurs uniques, {missing_pct:.1f}% manquantes")
            except Exception as e:
                print(f"  • {col}: Erreur lors de l'analyse - {str(e)}")
    
    return df_combined

def prepare_ml_data(df_combined):
    """
    Prépare les données pour le machine learning
    """
    print(f"\n🎯 PRÉPARATION POUR MACHINE LEARNING")
    print("="*70)
    
    # Variables cibles
    print("🎯 Création des variables cibles...")
    
    # Régression : prix
    if 'revenue' in df_combined.columns:
        y_regression = df_combined['revenue'].copy()
        print(f"  ✅ Régression: revenue (prix)")
        print(f"    • Min: ${float(y_regression.min()):.2f}")
        print(f"    • Max: ${float(y_regression.max()):.2f}")  
        print(f"    • Moyenne: ${float(y_regression.mean()):.2f}")
        print(f"    • Manquantes: {int(y_regression.isnull().sum())}")
    else:
        print("  ❌ Pas de variable revenue")
        y_regression = None
    
    # Classification : bien vs mal noté
    if 'sentiment_score' in df_combined.columns:
        sentiment_col = df_combined['sentiment_score']
        max_score = float(sentiment_col.max())
        
        if max_score <= 5:
            threshold = 4.5
            print(f"  ✅ Classification: sentiment >= {threshold} (échelle 1-5)")
        elif max_score <= 100:
            threshold = 80
            print(f"  ✅ Classification: sentiment >= {threshold} (échelle 1-100)")
        else:
            threshold = float(sentiment_col.median())
            print(f"  ✅ Classification: sentiment >= {threshold:.2f} (médiane)")
        
        y_classification = (sentiment_col >= threshold).astype(int)
        
        class_0 = int((y_classification == 0).sum())
        class_1 = int((y_classification == 1).sum())
        total = len(y_classification)
        
        print(f"    • Mal noté (0): {class_0:,} ({class_0/total*100:.1f}%)")
        print(f"    • Bien noté (1): {class_1:,} ({class_1/total*100:.1f}%)")
    else:
        print("  ❌ Pas de variable sentiment_score")
        y_classification = None
    
    # Features (X)
    print(f"\n🔧 Préparation des features...")
    exclude_cols = ['revenue', 'sentiment_score']
    feature_cols = [col for col in df_combined.columns if col not in exclude_cols]
    X = df_combined[feature_cols].copy()
    
    print(f"  📊 Features sélectionnées: {feature_cols}")
    
    # Encodage des variables catégorielles
    print(f"\n🔤 Encodage des variables catégorielles...")
    
    # Encodage city
    if 'city' in X.columns:
        le_city = LabelEncoder()
        X['city_encoded'] = le_city.fit_transform(X['city'].fillna('Unknown'))
        X = X.drop('city', axis=1)
        print(f"  ✅ City encodé: {int(X['city_encoded'].nunique())} valeurs")
    
    # Encodage day_of_week si c'est du texte
    if 'day_of_week' in X.columns:
        if X['day_of_week'].dtype == 'object':  # Si c'est du texte
            print(f"  🔤 day_of_week contient du texte, encodage...")
            le_dow = LabelEncoder()
            X['day_of_week'] = le_dow.fit_transform(X['day_of_week'].fillna('Unknown'))
            print(f"  ✅ day_of_week encodé: {int(X['day_of_week'].nunique())} valeurs")
        else:
            print(f"  ✅ day_of_week déjà numérique")
    
    # Vérification des types de données
    print(f"\n📊 Vérification des types:")
    for col in X.columns:
        dtype = X[col].dtype
        if dtype == 'object':
            print(f"  ⚠️  {col}: type object - encodage nécessaire")
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].fillna('Unknown'))
            print(f"    ✅ {col} encodé")
        else:
            print(f"  ✅ {col}: type {dtype}")
    
    # Gestion valeurs manquantes
    print(f"\n📊 Gestion des valeurs manquantes...")
    missing_before = X.isnull().sum().sum()
    
    if missing_before > 0:
        print(f"  ⚠️  {missing_before} valeurs manquantes détectées")
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(
            imputer.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        print(f"  ✅ Imputation terminée")
    else:
        print(f"  ✅ Aucune valeur manquante")
        X_imputed = X.copy()
    
    # Normalisation
    print(f"  🔧 Normalisation en cours...")
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_imputed),
        columns=X_imputed.columns,
        index=X_imputed.index
    )
    
    print(f"  ✅ Normalisation terminée")
    print(f"  ✅ Features finales: {list(X_scaled.columns)}")
    
    return X_scaled, y_regression, y_classification

def train_baseline_models(X, y_reg, y_clf):
    """
    Entraîne les modèles baseline
    """
    print(f"\n🤖 ENTRAÎNEMENT DES MODÈLES BASELINE")
    print("="*70)
    
    results = {}
    
    # RÉGRESSION
    if y_reg is not None and not y_reg.isnull().all():
        print(f"\n📊 RÉGRESSION - Prédiction des prix")
        print("-" * 50)
        
        # Nettoyage données régression
        mask_reg = ~y_reg.isnull()
        X_reg = X[mask_reg]
        y_reg_clean = y_reg[mask_reg]
        
        print(f"  📊 Données nettoyées: {len(X_reg):,} échantillons")
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_reg, y_reg_clean, test_size=0.2, random_state=42
        )
        
        print(f"  📊 Split: Train {len(X_train):,} | Test {len(X_test):,}")
        
        # Régression Linéaire
        print(f"  🔸 Régression Linéaire...")
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)
        
        rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
        r2_lr = r2_score(y_test, y_pred_lr)
        cv_lr = cross_val_score(lr, X_train, y_train, cv=5, scoring='r2')
        
        print(f"    • RMSE: ${rmse_lr:.2f}")
        print(f"    • R²: {r2_lr:.3f}")
        print(f"    • CV R²: {cv_lr.mean():.3f}±{cv_lr.std():.3f}")
        
        # Random Forest
        print(f"  🌲 Random Forest...")
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        
        rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
        r2_rf = r2_score(y_test, y_pred_rf)
        cv_rf = cross_val_score(rf, X_train, y_train, cv=5, scoring='r2')
        
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
        print(f"\n🎯 CLASSIFICATION - Bien noté vs Mal noté")
        print("-" * 50)
        
        # Nettoyage données classification
        mask_clf = ~y_clf.isnull()
        X_clf = X[mask_clf]
        y_clf_clean = y_clf[mask_clf]
        
        print(f"  📊 Données nettoyées: {len(X_clf):,} échantillons")
        
        if y_clf_clean.nunique() > 1:
            # Split stratifié
            X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
                X_clf, y_clf_clean, test_size=0.2, random_state=42, stratify=y_clf_clean
            )
            
            print(f"  📊 Split: Train {len(X_train_clf):,} | Test {len(X_test_clf):,}")
            
            # Random Forest Classifier
            print(f"  🌲 Random Forest Classifier...")
            rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf_clf.fit(X_train_clf, y_train_clf)
            y_pred_clf = rf_clf.predict(X_test_clf)
            
            accuracy = rf_clf.score(X_test_clf, y_test_clf)
            cv_clf = cross_val_score(rf_clf, X_train_clf, y_train_clf, cv=5, scoring='accuracy')
            
            print(f"    • Accuracy: {accuracy:.3f}")
            print(f"    • CV Accuracy: {cv_clf.mean():.3f}±{cv_clf.std():.3f}")
            
            # Matrice de confusion
            cm = confusion_matrix(y_test_clf, y_pred_clf)
            print(f"  📊 Matrice de confusion:")
            print(f"           Prédiction")
            print(f"    Réel    0     1")
            print(f"      0   {cm[0,0]:3d}   {cm[0,1]:3d}")
            print(f"      1   {cm[1,0]:3d}   {cm[1,1]:3d}")
            
            results['classification'] = {
                'accuracy': accuracy,
                'cv_mean': cv_clf.mean(),
                'cv_std': cv_clf.std(),
                'confusion_matrix': cm,
                'test_data': (X_test_clf, y_test_clf, y_pred_clf)
            }
        else:
            print(f"  ⚠️  Une seule classe détectée - classification impossible")
    
    return results

def create_visualizations(results):
    """
    Crée les visualisations des résultats
    """
    if not results:
        print("⚠️  Pas de résultats à visualiser")
        return
    
    print(f"\n📊 CRÉATION DES VISUALISATIONS")
    print("="*70)
    
    n_plots = len(results)
    if n_plots == 0:
        return
    
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # Graphique régression
    if 'regression' in results:
        ax = axes[plot_idx]
        X_test, y_test, y_pred_lr, y_pred_rf = results['regression']['test_data']
        
        lr_r2 = results['regression']['LinearRegression']['r2']
        rf_r2 = results['regression']['RandomForest']['r2']
        
        ax.scatter(y_test, y_pred_lr, alpha=0.6, label=f"Linear Reg (R²={lr_r2:.3f})", s=20)
        ax.scatter(y_test, y_pred_rf, alpha=0.6, label=f"Random Forest (R²={rf_r2:.3f})", s=20)
        
        min_val, max_val = y_test.min(), y_test.max()
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Prédiction parfaite')
        
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
        accuracy = results['classification']['accuracy']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Mal noté', 'Bien noté'],
                   yticklabels=['Mal noté', 'Bien noté'])
        ax.set_title(f"Classification (Accuracy: {accuracy:.3f})")
        ax.set_xlabel('Prédiction')
        ax.set_ylabel('Réel')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plots/ml_baseline_results.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Graphiques sauvegardés: {OUTPUT_DIR}/plots/ml_baseline_results.png")

def generate_report(results):
    """
    Génère le rapport final
    """
    print(f"\n📋 GÉNÉRATION DU RAPPORT FINAL")
    print("="*70)
    
    report_lines = []
    report_lines.append("# 📊 RAPPORT ML BASELINE - JOUR 3")
    report_lines.append("=" * 50)
    report_lines.append("")
    report_lines.append("## 🎯 RÉSUMÉ EXÉCUTIF")
    report_lines.append("")
    
    # Régression
    if 'regression' in results:
        lr_r2 = results['regression']['LinearRegression']['r2']
        rf_r2 = results['regression']['RandomForest']['r2']
        lr_rmse = results['regression']['LinearRegression']['rmse']
        rf_rmse = results['regression']['RandomForest']['rmse']
        
        report_lines.append("### 📊 Régression (Prédiction Prix)")
        report_lines.append("| Modèle | RMSE ($) | R² | CV R² |")
        report_lines.append("|--------|----------|----|----|")
        report_lines.append(f"| Linear Regression | ${lr_rmse:.2f} | {lr_r2:.3f} | {results['regression']['LinearRegression']['cv_mean']:.3f}±{results['regression']['LinearRegression']['cv_std']:.3f} |")
        report_lines.append(f"| Random Forest | ${rf_rmse:.2f} | {rf_r2:.3f} | {results['regression']['RandomForest']['cv_mean']:.3f}±{results['regression']['RandomForest']['cv_std']:.3f} |")
        report_lines.append("")
        
        best_reg_model = 'Random Forest' if rf_r2 > lr_r2 else 'Linear Regression'
        best_reg_r2 = max(rf_r2, lr_r2)
        report_lines.append(f"**🏆 Meilleur modèle**: {best_reg_model} (R² = {best_reg_r2:.3f})")
        report_lines.append("")
    
    # Classification
    if 'classification' in results:
        accuracy = results['classification']['accuracy']
        cv_mean = results['classification']['cv_mean']
        cv_std = results['classification']['cv_std']
        
        report_lines.append("### 🎯 Classification (Bien noté vs Mal noté)")
        report_lines.append("| Métrique | Valeur |")
        report_lines.append("|----------|--------|")
        report_lines.append(f"| Accuracy | {accuracy:.3f} |")
        report_lines.append(f"| CV Accuracy | {cv_mean:.3f}±{cv_std:.3f} |")
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
    
    return results

def main():
    """
    Fonction principale - Version définitivement corrigée
    """
    print("🚀 JOUR 3 - ML BASELINE (VERSION DÉFINITIVEMENT CORRIGÉE)")
    print("="*70)
    print("Objectif : Entraîner modèles baseline sans erreurs")
    print()
    
    try:
        # 1. Exploration et préparation
        df_combined = explore_and_prepare_data()
        
        # 2. Préparation ML
        X, y_regression, y_classification = prepare_ml_data(df_combined)
        
        # 3. Entraînement
        results = train_baseline_models(X, y_regression, y_classification)
        
        # 4. Visualisations
        create_visualizations(results)
        
        # 5. Rapport
        generate_report(results)
        
        # 6. Résumé final
        print(f"\n🏆 RÉSULTATS FINAUX")
        print("="*50)
        
        if 'regression' in results:
            lr_r2 = results['regression']['LinearRegression']['r2']
            rf_r2 = results['regression']['RandomForest']['r2']
            best_reg_r2 = max(lr_r2, rf_r2)
            best_reg_model = 'Random Forest' if rf_r2 > lr_r2 else 'Linear Regression'
            print(f"📊 RÉGRESSION: {best_reg_model} - R² = {best_reg_r2:.3f}")
        
        if 'classification' in results:
            accuracy = results['classification']['accuracy']
            print(f"🎯 CLASSIFICATION: Random Forest - Accuracy = {accuracy:.3f}")
        
        print("="*50)
        print(f"\n🎉 ML BASELINE TERMINÉ AVEC SUCCÈS !")
        print(f"📁 Résultats dans: {OUTPUT_DIR}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Erreur durant le ML Baseline: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()