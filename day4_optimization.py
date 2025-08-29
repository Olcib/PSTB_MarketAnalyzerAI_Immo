import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, VotingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression, RFE
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Configuration
PARIS_FILE = "data/processed/enriched_paris_fixed.csv"
SEATTLE_FILE = "data/processed/enriched_seattle_fixed.csv"
OUTPUT_DIR = "results/day4_optimization/"

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/plots/", exist_ok=True)

def load_enriched_features():
    """
    ÉTAPE 1: Chargement de TOUTES les features enrichies
    """
    print("ÉTAPE 1 - FEATURE ENGINEERING MASSIF")
    print("="*60)
    
    # Charger les datasets complets
    df_paris = pd.read_csv(PARIS_FILE)
    df_seattle = pd.read_csv(SEATTLE_FILE)
    
    print(f"Paris complet: {df_paris.shape[0]:,} lignes × {df_paris.shape[1]} colonnes")
    print(f"Seattle complet: {df_seattle.shape[0]:,} lignes × {df_seattle.shape[1]} colonnes")
    
    # Préparation des datasets avec plus de features
    def prepare_enhanced_data(df, city_name):
        df_enhanced = df.copy()
        df_enhanced['city'] = city_name
        
        # Features de base standardisées
        feature_mapping = {}
        
        if city_name == 'Paris' and 'estimated_occupancy_l365d' in df.columns:
            feature_mapping['estimated_occupancy_l365d'] = 'occupancy_rate'
        elif 'occupancy_rate' not in df.columns:
            df_enhanced['occupancy_rate'] = 50  # valeur par défaut
        
        if 'price' in df.columns:
            feature_mapping['price'] = 'revenue'
        
        if 'review_scores_rating' in df.columns:
            feature_mapping['review_scores_rating'] = 'sentiment_score'
        
        df_enhanced = df_enhanced.rename(columns=feature_mapping)
        
        return df_enhanced
    
    df_paris_enhanced = prepare_enhanced_data(df_paris, 'Paris')
    df_seattle_enhanced = prepare_enhanced_data(df_seattle, 'Seattle')
    
    # Identifier les colonnes communes importantes
    paris_cols = set(df_paris_enhanced.columns)
    seattle_cols = set(df_seattle_enhanced.columns)
    common_cols = paris_cols.intersection(seattle_cols)
    
    print(f"\nColonnes communes: {len(common_cols)}")
    
    # Sélectionner les features les plus pertinentes
    priority_features = []
    
    # Features de base
    base_features = ['occupancy_rate', 'revenue', 'sentiment_score', 'nb_amenities', 'month', 'day_of_week', 'city']
    for feature in base_features:
        if feature in common_cols:
            priority_features.append(feature)
    
    # Features géographiques si disponibles
    geo_features = ['latitude', 'longitude', 'neighbourhood_cleansed', 'neighbourhood_group_cleansed']
    for feature in geo_features:
        if feature in common_cols:
            priority_features.append(feature)
            print(f"  Géographique: {feature}")
    
    # Features de qualité
    quality_patterns = ['review_scores_', 'host_', 'property_type', 'room_type', 'bed_type']
    for col in common_cols:
        for pattern in quality_patterns:
            if pattern in col and col not in priority_features:
                priority_features.append(col)
    
    # Features temporelles enrichies (rolling, lag, etc.)
    temporal_patterns = ['_rolling_', '_lag_', 'season', 'quarter']
    for col in common_cols:
        for pattern in temporal_patterns:
            if pattern in col and col not in priority_features:
                priority_features.append(col)
    
    # Limiter à 50 features max pour éviter la surcharge
    priority_features = priority_features[:50]
    
    print(f"Features sélectionnées: {len(priority_features)}")
    print(f"Échantillon: {priority_features[:10]}")
    
    # Créer les datasets finaux
    df_paris_final = df_paris_enhanced[priority_features].copy()
    df_seattle_final = df_seattle_enhanced[priority_features].copy()
    
    # Combiner
    df_combined = pd.concat([df_paris_final, df_seattle_final], ignore_index=True)
    
    print(f"\nDataset enrichi final: {df_combined.shape[0]:,} × {df_combined.shape[1]}")
    
    return df_combined

def fix_classification_target(df_combined):
    """
    ÉTAPE 2: Correction du problème de classification déséquilibrée
    """
    print("\nÉTAPE 2 - CORRECTION CLASSIFICATION")
    print("="*50)
    
    if 'sentiment_score' not in df_combined.columns:
        print("Pas de variable sentiment_score disponible")
        return df_combined, None, None
    
    sentiment_col = df_combined['sentiment_score'].dropna()
    
    # Analyser la distribution
    print(f"Distribution sentiment_score:")
    print(f"  Min: {sentiment_col.min():.2f}")
    print(f"  Max: {sentiment_col.max():.2f}")
    print(f"  Médiane: {sentiment_col.median():.2f}")
    print(f"  Percentile 75: {sentiment_col.quantile(0.75):.2f}")
    
    # Nouveau seuil pour équilibrer les classes (percentile 70)
    new_threshold = sentiment_col.quantile(0.70)
    
    y_classification_balanced = (df_combined['sentiment_score'] >= new_threshold).astype(int)
    
    # Vérifier le nouvel équilibre
    class_counts = y_classification_balanced.value_counts()
    total = len(y_classification_balanced)
    
    print(f"\nNouveau seuil: >= {new_threshold:.2f}")
    print(f"  Mal noté (0): {class_counts.get(0, 0):,} ({class_counts.get(0, 0)/total*100:.1f}%)")
    print(f"  Bien noté (1): {class_counts.get(1, 0):,} ({class_counts.get(1, 0)/total*100:.1f}%)")
    
    # Variable de régression
    y_regression = df_combined['revenue'].copy() if 'revenue' in df_combined.columns else None
    
    return df_combined, y_regression, y_classification_balanced

def prepare_features_advanced(df_combined):
    """
    Préparation avancée des features avec nettoyage robuste
    """
    print("\nPréparation features avancées...")
    
    # Exclure les variables cibles
    exclude_cols = ['revenue', 'sentiment_score']
    feature_cols = [col for col in df_combined.columns if col not in exclude_cols]
    X = df_combined[feature_cols].copy()
    
    # Nettoyage des colonnes
    for col in X.columns:
        if X[col].dtype == 'object':
            # Encoder les variables catégorielles
            if col == 'city':
                le = LabelEncoder()
                X[col + '_encoded'] = le.fit_transform(X[col].fillna('Unknown'))
                X = X.drop(col, axis=1)
            else:
                # Autres variables catégorielles
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str).fillna('Unknown'))
        else:
            # Variables numériques
            X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Imputation des valeurs manquantes
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    
    # Normalisation
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_imputed),
        columns=X_imputed.columns,
        index=X_imputed.index
    )
    
    print(f"Features finales: {X_scaled.shape[1]} colonnes")
    
    return X_scaled

def advanced_feature_selection(X, y):
    """
    ÉTAPE 3: Sélection intelligente des features
    """
    print("\nÉTAPE 3 - SÉLECTION DE FEATURES")
    print("="*45)
    
    if y is None or len(X) == 0:
        return X
    
    # Nettoyer y
    y_clean = pd.to_numeric(y, errors='coerce').fillna(y.median() if hasattr(y, 'median') else 0)
    
    # Sélection des K meilleures features
    k = min(20, X.shape[1])  # Max 20 features
    
    selector = SelectKBest(f_regression, k=k)
    X_selected = selector.fit_transform(X, y_clean)
    
    # Récupérer les noms des features sélectionnées
    selected_features = X.columns[selector.get_support()].tolist()
    
    X_final = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    print(f"Features sélectionnées: {len(selected_features)}")
    print(f"Top 10: {selected_features[:10]}")
    
    return X_final

def train_advanced_models(X, y_reg, y_clf):
    """
    ÉTAPE 4: Entraînement des modèles avancés
    """
    print("\nÉTAPE 4 - MODÈLES AVANCÉS")
    print("="*35)
    
    results = {}
    
    # RÉGRESSION AVANCÉE
    if y_reg is not None and not y_reg.isnull().all():
        print("\nRégression avec modèles avancés...")
        
        # Nettoyage
        mask_reg = ~y_reg.isnull()
        X_reg = X[mask_reg]
        y_reg_clean = y_reg[mask_reg]
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_reg, y_reg_clean, test_size=0.2, random_state=42
        )
        
        print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
        
        # Modèles à tester
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=200, max_depth=8, learning_rate=0.1, random_state=42),
            'LightGBM': lgb.LGBMRegressor(n_estimators=200, max_depth=8, learning_rate=0.1, random_state=42, verbose=-1)
        }
        
        regression_results = {}
        
        for name, model in models.items():
            print(f"\n{name}...")
            
            try:
                # Entraînement
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Métriques
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
                # Validation croisée
                cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='r2')
                
                print(f"  RMSE: ${rmse:.2f}")
                print(f"  R²: {r2:.3f}")
                print(f"  CV R²: {cv_scores.mean():.3f}±{cv_scores.std():.3f}")
                
                regression_results[name] = {
                    'rmse': rmse,
                    'r2': r2,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'predictions': y_pred
                }
                
            except Exception as e:
                print(f"  Erreur avec {name}: {str(e)}")
        
        if regression_results:
            results['regression'] = {
                'models': regression_results,
                'test_data': (X_test, y_test)
            }
    
    # CLASSIFICATION AVANCÉE
    if y_clf is not None and not y_clf.isnull().all():
        print("\nClassification avec Random Forest optimisé...")
        
        # Nettoyage
        mask_clf = ~y_clf.isnull()
        X_clf = X[mask_clf]
        y_clf_clean = y_clf[mask_clf]
        
        if y_clf_clean.nunique() > 1:
            # Split
            X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
                X_clf, y_clf_clean, test_size=0.2, random_state=42, stratify=y_clf_clean
            )
            
            # Random Forest optimisé
            rf_clf = RandomForestClassifier(
                n_estimators=200, 
                max_depth=10, 
                class_weight='balanced',  # Important pour l'équilibrage
                random_state=42
            )
            
            rf_clf.fit(X_train_clf, y_train_clf)
            y_pred_clf = rf_clf.predict(X_test_clf)
            
            # Métriques
            accuracy = rf_clf.score(X_test_clf, y_test_clf)
            cv_scores = cross_val_score(rf_clf, X_train_clf, y_train_clf, cv=3, scoring='accuracy')
            
            print(f"  Accuracy: {accuracy:.3f}")
            print(f"  CV Accuracy: {cv_scores.mean():.3f}±{cv_scores.std():.3f}")
            
            # Matrice de confusion
            cm = confusion_matrix(y_test_clf, y_pred_clf)
            print(f"  Matrice de confusion:")
            print(f"           Prédiction")
            print(f"    Réel    0     1")
            print(f"      0   {cm[0,0]:3d}   {cm[0,1]:3d}")
            print(f"      1   {cm[1,0]:3d}   {cm[1,1]:3d}")
            
            results['classification'] = {
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'confusion_matrix': cm,
                'test_data': (X_test_clf, y_test_clf, y_pred_clf)
            }
    
    return results

def create_comparison_plots(results, baseline_results=None):
    """
    ÉTAPE 5: Comparaison avec le baseline
    """
    print("\nÉTAPE 5 - COMPARAISON VISUELLE")
    print("="*40)
    
    # Baseline du Jour 3 (hardcodé car on connaît les résultats)
    baseline_r2 = 0.066
    baseline_accuracy = 0.985
    
    if 'regression' in results:
        # Trouver le meilleur modèle
        best_model = max(results['regression']['models'], 
                        key=lambda x: results['regression']['models'][x]['r2'])
        best_r2 = results['regression']['models'][best_model]['r2']
        
        print(f"\nAMÉLIORAIRE RÉGRESSION:")
        print(f"  Baseline (Jour 3): R² = {baseline_r2:.3f}")
        print(f"  Meilleur (Jour 4): R² = {best_r2:.3f} ({best_model})")
        improvement = ((best_r2 - baseline_r2) / abs(baseline_r2)) * 100
        print(f"  Amélioration: {improvement:+.1f}%")
        
        # Évaluation du succès
        if best_r2 > 0.3:
            print("  ✅ SUCCÈS: R² > 0.3 (objectif atteint)")
        elif best_r2 > baseline_r2 * 2:
            print("  🔶 PROGRÈS: Amélioration significative")
        else:
            print("  ❌ ÉCHEC: Amélioration insuffisante")
    
    if 'classification' in results:
        new_accuracy = results['classification']['accuracy']
        
        print(f"\nAMÉLIORATION CLASSIFICATION:")
        print(f"  Baseline (Jour 3): Accuracy = {baseline_accuracy:.3f} (artificielle)")
        print(f"  Équilibrée (Jour 4): Accuracy = {new_accuracy:.3f} (réelle)")
        
        # La comparaison n'est pas directe car on a changé le problème
        if new_accuracy > 0.75:
            print("  ✅ SUCCÈS: Classification équilibrée > 75%")
        elif new_accuracy > 0.60:
            print("  🔶 CORRECT: Performance acceptable")
        else:
            print("  ❌ FAIBLE: Classification encore difficile")

def generate_day4_report(results):
    """
    Génération du rapport final Jour 4
    """
    print(f"\nRAPPORT FINAL JOUR 4")
    print("="*30)
    
    # Créer le rapport
    report_lines = []
    report_lines.append("# RAPPORT JOUR 4 - OPTIMISATION ML")
    report_lines.append("="*50)
    report_lines.append("")
    
    # Résumé des améliorations
    if 'regression' in results:
        best_model = max(results['regression']['models'], 
                        key=lambda x: results['regression']['models'][x]['r2'])
        best_r2 = results['regression']['models'][best_model]['r2']
        
        report_lines.append("## RÉGRESSION")
        report_lines.append(f"- **Baseline Jour 3**: R² = 0.066")
        report_lines.append(f"- **Meilleur Jour 4**: R² = {best_r2:.3f} ({best_model})")
        
        improvement = ((best_r2 - 0.066) / 0.066) * 100
        report_lines.append(f"- **Amélioration**: {improvement:+.1f}%")
        report_lines.append("")
        
        # Détails des modèles
        for name, metrics in results['regression']['models'].items():
            report_lines.append(f"### {name}")
            report_lines.append(f"- RMSE: ${metrics['rmse']:.2f}")
            report_lines.append(f"- R²: {metrics['r2']:.3f}")
            report_lines.append(f"- CV: {metrics['cv_mean']:.3f}±{metrics['cv_std']:.3f}")
            report_lines.append("")
    
    if 'classification' in results:
        accuracy = results['classification']['accuracy']
        
        report_lines.append("## CLASSIFICATION")
        report_lines.append(f"- **Jour 3**: 98.5% (artificiel, déséquilibré)")
        report_lines.append(f"- **Jour 4**: {accuracy:.3f} (équilibré)")
        report_lines.append("")
    
    # Conclusions
    report_lines.append("## CONCLUSIONS")
    if 'regression' in results:
        best_r2 = max(results['regression']['models'].values(), key=lambda x: x['r2'])['r2']
        if best_r2 > 0.3:
            report_lines.append("✅ **Succès**: Objectifs atteints")
        else:
            report_lines.append("🔶 **Progrès**: Améliorations mais objectifs non atteints")
    
    # Sauvegarder
    with open(f"{OUTPUT_DIR}/jour4_optimization_report.md", 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✅ Rapport: {OUTPUT_DIR}/jour4_optimization_report.md")

def main():
    """
    Workflow principal Jour 4
    """
    print("🚀 JOUR 4 - OPTIMISATION ML AVANCÉE")
    print("="*60)
    print("Objectif: Améliorer drastiquement les performances du Baseline")
    print()
    
    try:
        # Étape 1: Features enrichies
        df_combined = load_enriched_features()
        
        # Étape 2: Correction classification
        df_combined, y_regression, y_classification = fix_classification_target(df_combined)
        
        # Préparation des features
        X = prepare_features_advanced(df_combined)
        
        # Étape 3: Sélection des features
        if y_regression is not None:
            X = advanced_feature_selection(X, y_regression)
        
        # Étape 4: Modèles avancés
        results = train_advanced_models(X, y_regression, y_classification)
        
        # Étape 5: Comparaison et rapport
        create_comparison_plots(results)
        generate_day4_report(results)
        
        print("\n🏆 JOUR 4 TERMINÉ")
        print(f"📁 Résultats: {OUTPUT_DIR}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Erreur Jour 4: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()