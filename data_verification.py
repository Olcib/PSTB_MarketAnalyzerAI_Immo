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

def load_and_prepare_data():
    """
    Charge et prépare les données pour ML
    """
    print("📊 CHARGEMENT DES DONNÉES")
    print("="*50)
    
    # Chargement
    df_paris = pd.read_csv(PARIS_FILE)
    df_seattle = pd.read_csv(SEATTLE_FILE)
    
    print(f"✅ Paris : {df_paris.shape[0]:,} lignes × {df_paris.shape[1]} colonnes")
    print(f"✅ Seattle : {df_seattle.shape[0]:,} lignes × {df_seattle.shape[1]} colonnes")
    
    # Harmonisation des colonnes occupancy_rate
    if 'estimated_occupancy_l365d' in df_paris.columns:
        df_paris['occupancy_rate'] = df_paris['estimated_occupancy_l365d']
    
    # Combinaison des datasets
    df_combined = pd.concat([df_paris, df_seattle], ignore_index=True)
    print(f"✅ Dataset combiné : {df_combined.shape[0]:,} lignes × {df_combined.shape[1]} colonnes")
    
    return df_paris, df_seattle, df_combined

def create_baseline_features(df):
    """
    Crée les features de base pour ML Baseline
    """
    print("\n🎯 CRÉATION DES FEATURES DE BASE")
    print("="*50)
    
    # Features de base requises
    base_features = [
        'occupancy_rate', 'price', 'review_scores_rating', 
        'nb_amenities', 'day_of_week', 'month'
    ]
    
    # Vérification de la présence des features
    missing_features = [f for f in base_features if f not in df.columns]
    if missing_features:
        print(f"⚠️  Features manquantes : {missing_features}")
        return None
    
    # Sélection et nettoyage
    df_ml = df[base_features + ['city']].copy()
    
    # Renommage pour clarté
    feature_mapping = {
        'price': 'revenue',
        'review_scores_rating': 'sentiment_score'
    }
    df_ml = df_ml.rename(columns=feature_mapping)
    
    print("✅ Features de base créées :")
    for feature in ['occupancy_rate', 'revenue', 'sentiment_score', 'nb_amenities', 'day_of_week', 'month']:
        if feature in df_ml.columns:
            missing_pct = df_ml[feature].isnull().sum() / len(df_ml) * 100
            unique_vals = df_ml[feature].nunique()
            print(f"  • {feature}: {unique_vals} valeurs uniques, {missing_pct:.1f}% manquantes")
    
    return df_ml

def prepare_targets(df_ml):
    """
    Prépare les variables cibles pour régression et classification
    """
    print("\n🎯 PRÉPARATION DES VARIABLES CIBLES")
    print("="*50)
    
    # Variable cible régression : revenue (price)
    y_regression = df_ml['revenue'].copy()
    
    # Variable cible classification : "bien noté" vs "mal noté"
    # Seuil : >= 4.5/5 pour Paris, >= 80/100 pour Seattle  
    threshold_condition = (
        ((df_ml['sentiment_score'] >= 4.5) & (df_ml['sentiment_score'] <= 5)) |
        ((df_ml['sentiment_score'] >= 80) & (df_ml['sentiment_score'] <= 100))
    )
    
    y_classification = threshold_condition.astype(int)
    
    print(f"📊 Régression (revenue) :")
    print(f"  • Min: ${y_regression.min():.2f}")
    print(f"  • Max: ${y_regression.max():.2f}")
    print(f"  • Moyenne: ${y_regression.mean():.2f}")
    print(f"  • Valeurs manquantes: {y_regression.isnull().sum()}")
    
    print(f"\n📊 Classification (bien noté) :")
    class_counts = y_classification.value_counts()
    print(f"  • Mal noté (0): {class_counts.get(0, 0):,} ({class_counts.get(0, 0)/len(y_classification)*100:.1f}%)")
    print(f"  • Bien noté (1): {class_counts.get(1, 0):,} ({class_counts.get(1, 0)/len(y_classification)*100:.1f}%)")
    
    return y_regression, y_classification

def preprocess_features(df_ml):
    """
    Préprocessing des features pour ML
    """
    print("\n🔧 PREPROCESSING DES FEATURES")
    print("="*50)
    
    # Copie pour preprocessing
    X = df_ml.drop(['revenue', 'sentiment_score'], axis=1, errors='ignore').copy()
    
    # Encodage de la variable city
    if 'city' in X.columns:
        le_city = LabelEncoder()
        X['city_encoded'] = le_city.fit_transform(X['city'].fillna('Unknown'))
        X = X.drop('city', axis=1)
        print(f"✅ City encodé : {X['city_encoded'].nunique()} valeurs")
    
    # Gestion des valeurs manquantes
    print(f"📊 Valeurs manquantes avant imputation :")
    missing_before = X.isnull().sum()
    for col, missing in missing_before.items():
        if missing > 0:
            print(f"  • {col}: {missing} ({missing/len(X)*100:.1f}%)")
    
    # Imputation
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X), 
        columns=X.columns, 
        index=X.index
    )
    
    missing_after = X_imputed.isnull().sum().sum()
    print(f"✅ Valeurs manquantes après imputation : {missing_after}")
    
    # Normalisation
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_imputed), 
        columns=X_imputed.columns, 
        index=X_imputed.index
    )
    
    print(f"✅ Features normalisées : {X_scaled.shape[1]} colonnes")
    
    return X_scaled, imputer, scaler

def train_regression_models(X, y):
    """
    Entraîne les modèles de régression baseline
    """
    print("\n🤖 ENTRAÎNEMENT MODÈLES RÉGRESSION")
    print("="*50)
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"📊 Split données :")
    print(f"  • Train: {X_train.shape[0]:,} échantillons")
    print(f"  • Test: {X_test.shape[0]:,} échantillons")
    
    results = {}
    
    # 1. Régression Linéaire
    print(f"\n🔸 Régression Linéaire...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    # Prédictions
    y_pred_lr = lr.predict(X_test)
    
    # Métriques
    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    r2_lr = r2_score(y_test, y_pred_lr)
    
    # Validation croisée
    cv_scores_lr = cross_val_score(lr, X_train, y_train, cv=5, scoring='r2')
    
    results['Linear Regression'] = {
        'model': lr,
        'y_pred': y_pred_lr,
        'rmse': rmse_lr,
        'r2': r2_lr,
        'cv_r2_mean': cv_scores_lr.mean(),
        'cv_r2_std': cv_scores_lr.std()
    }
    
    print(f"  ✅ RMSE: ${rmse_lr:.2f}")
    print(f"  ✅ R²: {r2_lr:.3f}")
    print(f"  ✅ CV R² (mean±std): {cv_scores_lr.mean():.3f}±{cv_scores_lr.std():.3f}")
    
    # 2. Random Forest
    print(f"\n🌲 Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # Prédictions
    y_pred_rf = rf.predict(X_test)
    
    # Métriques
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    r2_rf = r2_score(y_test, y_pred_rf)
    
    # Validation croisée
    cv_scores_rf = cross_val_score(rf, X_train, y_train, cv=5, scoring='r2')
    
    results['Random Forest'] = {
        'model': rf,
        'y_pred': y_pred_rf,
        'rmse': rmse_rf,
        'r2': r2_rf,
        'cv_r2_mean': cv_scores_rf.mean(),
        'cv_r2_std': cv_scores_rf.std()
    }
    
    print(f"  ✅ RMSE: ${rmse_rf:.2f}")
    print(f"  ✅ R²: {r2_rf:.3f}")
    print(f"  ✅ CV R² (mean±std): {cv_scores_rf.mean():.3f}±{cv_scores_rf.std():.3f}")
    
    return results, X_test, y_test

def train_classification_model(X, y):
    """
    Entraîne le modèle de classification baseline
    """
    print("\n🎯 ENTRAÎNEMENT MODÈLE CLASSIFICATION")
    print("="*50)
    
    # Supprimer les lignes avec target manquante
    mask = ~y.isnull()
    X_clean = X[mask]
    y_clean = y[mask]
    
    print(f"📊 Données nettoyées : {len(X_clean):,} échantillons")
    
    # Split train/test stratifié
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean
    )
    
    # Random Forest Classifier
    print(f"\n🌲 Random Forest Classifier...")
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_clf.fit(X_train, y_train)
    
    # Prédictions
    y_pred_clf = rf_clf.predict(X_test)
    y_pred_proba = rf_clf.predict_proba(X_test)[:, 1]
    
    # Validation croisée stratifiée
    cv_scores = cross_val_score(
        rf_clf, X_train, y_train, 
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='accuracy'
    )
    
    print(f"  ✅ Accuracy: {rf_clf.score(X_test, y_test):.3f}")
    print(f"  ✅ CV Accuracy (mean±std): {cv_scores.mean():.3f}±{cv_scores.std():.3f}")
    
    return rf_clf, X_test, y_test, y_pred_clf, y_pred_proba

def create_evaluation_plots(regression_results, X_test_reg, y_test_reg, 
                          clf_model, X_test_clf, y_test_clf, y_pred_clf):
    """
    Crée les graphiques d'évaluation
    """
    print("\n📊 CRÉATION DES GRAPHIQUES D'ÉVALUATION")
    print("="*50)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Régression - Prédictions vs Réel
    ax1 = axes[0, 0]
    for model_name, results in regression_results.items():
        y_pred = results['y_pred']
        ax1.scatter(y_test_reg, y_pred, alpha=0.5, label=f"{model_name} (R²={results['r2']:.3f})")
    
    # Ligne parfaite
    min_val, max_val = y_test_reg.min(), y_test_reg.max()
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', label='Prédiction parfaite')
    
    ax1.set_xlabel('Prix Réel ($)')
    ax1.set_ylabel('Prix Prédit ($)')
    ax1.set_title('Régression: Prédictions vs Réel')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Régression - Résidus
    ax2 = axes[0, 1]
    for model_name, results in regression_results.items():
        residuals = y_test_reg - results['y_pred']
        ax2.scatter(results['y_pred'], residuals, alpha=0.5, label=model_name)
    
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel('Prix Prédit ($)')
    ax2.set_ylabel('Résidus ($)')
    ax2.set_title('Régression: Analyse des Résidus')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Classification - Matrice de confusion
    ax3 = axes[1, 0]
    cm = confusion_matrix(y_test_clf, y_pred_clf)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
                xticklabels=['Mal noté', 'Bien noté'],
                yticklabels=['Mal noté', 'Bien noté'])
    ax3.set_title('Classification: Matrice de Confusion')
    ax3.set_xlabel('Prédiction')
    ax3.set_ylabel('Réel')
    
    # 4. Feature Importance (Random Forest)
    ax4 = axes[1, 1]
    rf_reg = regression_results['Random Forest']['model']
    feature_importance = pd.DataFrame({
        'feature': X_test_reg.columns,
        'importance': rf_reg.feature_importances_
    }).sort_values('importance', ascending=True)
    
    ax4.barh(feature_importance['feature'], feature_importance['importance'])
    ax4.set_title('Random Forest: Importance des Features')
    ax4.set_xlabel('Importance')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plots/ml_baseline_evaluation.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Graphiques sauvegardés dans {OUTPUT_DIR}/plots/")

def generate_report(regression_results, clf_results, y_test_clf, y_pred_clf):
    """
    Génère le rapport final
    """
    print("\n📋 GÉNÉRATION DU RAPPORT FINAL")
    print("="*50)
    
    report_content = []
    report_content.append("# 📊 RAPPORT ML BASELINE - JOUR 3")
    report_content.append("=" * 50)
    report_content.append("")
    
    # Résumé exécutif
    report_content.append("## 🎯 RÉSUMÉ EXÉCUTIF")
    report_content.append("")
    report_content.append("### 📊 Performances des Modèles")
    report_content.append("")
    
    # Tableau régression
    report_content.append("#### Régression (Prédiction Prix)")
    report_content.append("| Modèle | RMSE ($) | R² | CV R² (mean±std) |")
    report_content.append("|--------|----------|----|--------------------|")
    
    for model_name, results in regression_results.items():
        rmse = results['rmse']
        r2 = results['r2']
        cv_mean = results['cv_r2_mean']
        cv_std = results['cv_r2_std']
        report_content.append(f"| {model_name} | ${rmse:.2f} | {r2:.3f} | {cv_mean:.3f}±{cv_std:.3f} |")
    
    # Classification
    report_content.append("")
    report_content.append("#### Classification (Bien noté vs Mal noté)")
    
    # Matrice de confusion
    cm = confusion_matrix(y_test_clf, y_pred_clf)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    report_content.append("")
    report_content.append("| Métrique | Valeur |")
    report_content.append("|----------|--------|")
    report_content.append(f"| Accuracy | {accuracy:.3f} |")
    report_content.append(f"| Precision | {precision:.3f} |")
    report_content.append(f"| Recall | {recall:.3f} |")
    report_content.append(f"| F1-Score | {f1:.3f} |")
    
    report_content.append("")
    report_content.append("### 📊 Matrice de Confusion")
    report_content.append("```")
    report_content.append(f"                Prédiction")
    report_content.append(f"Réel      Mal noté  Bien noté")
    report_content.append(f"Mal noté     {tn:4d}     {fp:4d}")
    report_content.append(f"Bien noté    {fn:4d}     {tp:4d}")
    report_content.append("```")
    
    # Conclusions
    report_content.append("")
    report_content.append("## 🎯 CONCLUSIONS")
    report_content.append("")
    
    # Meilleur modèle régression
    best_reg_model = max(regression_results.keys(), 
                        key=lambda x: regression_results[x]['r2'])
    best_r2 = regression_results[best_reg_model]['r2']
    
    report_content.append(f"### ✅ Régression")
    report_content.append(f"- **Meilleur modèle** : {best_reg_model}")
    report_content.append(f"- **Performance** : R² = {best_r2:.3f}")
    
    if best_r2 > 0.7:
        report_content.append("- **Qualité** : Excellente prédiction")
    elif best_r2 > 0.5:
        report_content.append("- **Qualité** : Bonne prédiction")
    else:
        report_content.append("- **Qualité** : Prédiction modérée")
    
    report_content.append("")
    report_content.append(f"### ✅ Classification")
    report_content.append(f"- **Accuracy** : {accuracy:.3f}")
    
    if accuracy > 0.8:
        report_content.append("- **Qualité** : Excellente classification")
    elif accuracy > 0.7:
        report_content.append("- **Qualité** : Bonne classification")
    else:
        report_content.append("- **Qualité** : Classification modérée")
    
    # Sauvegarde du rapport
    with open(f"{OUTPUT_DIR}/ml_baseline_report.md", 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_content))
    
    print(f"✅ Rapport sauvegardé : {OUTPUT_DIR}/ml_baseline_report.md")
    
    # Affichage console
    print("\n" + "="*70)
    print("🏆 RÉSULTATS ML BASELINE")
    print("="*70)
    print(f"📊 RÉGRESSION - Meilleur: {best_reg_model} (R² = {best_r2:.3f})")
    print(f"🎯 CLASSIFICATION - Accuracy: {accuracy:.3f}")
    print("="*70)

def main():
    """
    Fonction principale ML Baseline
    """
    print("🚀 JOUR 3 - ML BASELINE")
    print("="*70)
    print("Objectif : Entraîner modèles baseline et évaluer performances")
    print()
    
    try:
        # 1. Chargement des données
        df_paris, df_seattle, df_combined = load_and_prepare_data()
        
        # 2. Création des features de base
        df_ml = create_baseline_features(df_combined)
        if df_ml is None:
            return
        
        # 3. Préparation des targets
        y_regression, y_classification = prepare_targets(df_ml)
        
        # 4. Preprocessing
        X, imputer, scaler = preprocess_features(df_ml)
        
        # 5. Entraînement régression
        regression_results, X_test_reg, y_test_reg = train_regression_models(X, y_regression)
        
        # 6. Entraînement classification
        clf_model, X_test_clf, y_test_clf, y_pred_clf, y_pred_proba = train_classification_model(X, y_classification)
        
        # 7. Évaluation graphique
        create_evaluation_plots(regression_results, X_test_reg, y_test_reg,
                              clf_model, X_test_clf, y_test_clf, y_pred_clf)
        
        # 8. Rapport final
        clf_results = {
            'accuracy': clf_model.score(X_test_clf, y_test_clf),
            'predictions': y_pred_clf,
            'probabilities': y_pred_proba
        }
        
        generate_report(regression_results, clf_results, y_test_clf, y_pred_clf)
        
        print("\n🎉 ML BASELINE TERMINÉ AVEC SUCCÈS !")
        print(f"📁 Résultats dans : {OUTPUT_DIR}")
        
    except Exception as e:
        print(f"\n❌ Erreur ML Baseline : {str(e)}")
        raise

if __name__ == "__main__":
    main()