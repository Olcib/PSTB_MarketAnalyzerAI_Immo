import pandas as pd
import numpy as np
import os  
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Configuration
OUTPUT_DIR = "results/honest_baseline/"
MODEL_DIR = "models/honest_baseline/"

from pathlib import Path
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(f"{OUTPUT_DIR}/plots/").mkdir(parents=True, exist_ok=True)
Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

def load_clean_datasets():
    """
    Charge les datasets propres créés par l'Action 2
    """
    print("MODELE BASELINE HONNETE - ACTION 3")
    print("="*45)
    
    datasets = {}
    
    try:
        df_paris = pd.read_csv("results/clean_dataset/paris_clean_dataset.csv")
        datasets['Paris'] = df_paris
        print(f"Paris propre chargé: {df_paris.shape}")
    except FileNotFoundError:
        print("Dataset Paris propre non trouvé")
    
    try:
        df_seattle = pd.read_csv("results/clean_dataset/seattle_clean_dataset.csv")
        datasets['Seattle'] = df_seattle
        print(f"Seattle propre chargé: {df_seattle.shape}")
    except FileNotFoundError:
        print("Dataset Seattle propre non trouvé")
    
    return datasets

def handle_high_cardinality_features(X, max_categories=50):
    """
    Gère les variables catégorielles à cardinalité excessive
    """
    print(f"\nGESTION CARDINALITE EXCESSIVE")
    print("="*35)
    
    high_cardinality_cols = []
    
    for col in X.columns:
        if X[col].dtype in ['int64', 'float64']:
            unique_count = X[col].nunique()
            if unique_count > max_categories and col not in ['latitude', 'longitude', 'price']:
                high_cardinality_cols.append((col, unique_count))
    
    if high_cardinality_cols:
        print("Variables à cardinalité excessive détectées:")
        for col, count in high_cardinality_cols:
            print(f"  {col}: {count:,} catégories")
        
        # Stratégies de réduction
        X_processed = X.copy()
        
        for col, count in high_cardinality_cols:
            if 'amenities' in col.lower():
                # Pour amenities : garder seulement top 20 + regrouper autres
                top_amenities = X[col].value_counts().head(20).index
                X_processed[col] = X[col].apply(lambda x: x if x in top_amenities else 'other')
                print(f"  {col} réduit à 21 catégories (top 20 + other)")
            
            elif 'neighbourhood' in col.lower():
                # Pour quartiers : garder top 15 + regrouper
                top_neighbourhoods = X[col].value_counts().head(15).index
                X_processed[col] = X[col].apply(lambda x: x if x in top_neighbourhoods else 'other')
                print(f"  {col} réduit à 16 catégories (top 15 + other)")
            
            else:
                # Autres : regrouper par fréquence
                top_values = X[col].value_counts().head(10).index
                X_processed[col] = X[col].apply(lambda x: x if x in top_values else 'other')
                print(f"  {col} réduit à 11 catégories (top 10 + other)")
        
        return X_processed
    else:
        print("Aucune cardinalité excessive détectée")
        return X

def prepare_modeling_data(df, city_name):
    """
    Prépare les données pour la modélisation
    """
    print(f"\nPREPARATION MODELISATION - {city_name.upper()}")
    print("="*40)
    
    # Séparer features et target
    if 'price' not in df.columns:
        print(f"Variable price manquante pour {city_name}")
        return None, None, None, None
    
    X = df.drop('price', axis=1).copy()
    y = df['price'].copy()
    
    print(f"Features initiales: {X.shape[1]}")
    print(f"Échantillons: {len(X):,}")
    
    # Gérer cardinalité excessive
    X_processed = handle_high_cardinality_features(X)
    
    # Réencodage nécessaire après regroupement
    from sklearn.preprocessing import LabelEncoder
    for col in X_processed.columns:
        if X_processed[col].dtype == 'object':
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X_processed[col].astype(str).fillna('unknown'))

    # Sélection features par importance statistique
    print(f"\nSélection des features les plus prédictives...")
    
    # Limiter à 15 features max pour éviter overfitting
    k_best = min(15, X_processed.shape[1])
    
    try:
        selector = SelectKBest(f_regression, k=k_best)
        X_selected = selector.fit_transform(X_processed, y)
        
        selected_features = X_processed.columns[selector.get_support()]
        X_final = pd.DataFrame(X_selected, columns=selected_features)
        
        print(f"Features sélectionnées: {len(selected_features)}")
        
        # Afficher scores
        feature_scores = pd.DataFrame({
            'feature': X_processed.columns,
            'score': selector.scores_
        }).sort_values('score', ascending=False)
        
        print("Top 10 features par importance:")
        for i, (_, row) in enumerate(feature_scores.head(10).iterrows(), 1):
            selected = "✓" if row['feature'] in selected_features else " "
            print(f"  {i:2d}. {selected} {row['feature']}: {row['score']:.1f}")
        
    except Exception as e:
        print(f"Erreur sélection features: {e}")
        X_final = X_processed
        selected_features = X_processed.columns
    
    # Split train/test stratifié
    try:
        # Stratification par quartiles de prix
        price_quartiles = pd.qcut(y, q=4, labels=False, duplicates='drop')
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_final, y, test_size=0.2, random_state=42, 
            stratify=price_quartiles
        )
    except:
        # Fallback sans stratification si problème
        X_train, X_test, y_train, y_test = train_test_split(
            X_final, y, test_size=0.2, random_state=42
        )
    
    print(f"Split final: Train {len(X_train):,} | Test {len(X_test):,}")
    
    return X_train, X_test, y_train, y_test

def train_baseline_models(X_train, X_test, y_train, y_test, city_name):
    """
    Entraîne les modèles baseline honnêtes
    """
    print(f"\nENTRAINEMENT MODELES BASELINE - {city_name.upper()}")
    print("="*45)
    
    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {}
    results = {}
    
    # 1. Régression Linéaire
    print("1. Régression Linéaire...")
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    
    y_pred_lr = lr.predict(X_test_scaled)
    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    r2_lr = r2_score(y_test, y_pred_lr)
    
    # Cross-validation
    cv_scores_lr = cross_val_score(lr, X_train_scaled, y_train, cv=5, scoring='r2')
    
    print(f"   RMSE: ${rmse_lr:.2f}")
    print(f"   R²: {r2_lr:.3f}")
    print(f"   CV R²: {cv_scores_lr.mean():.3f}±{cv_scores_lr.std():.3f}")
    
    models['LinearRegression'] = lr
    results['LinearRegression'] = {
        'rmse': rmse_lr, 'r2': r2_lr, 
        'cv_mean': cv_scores_lr.mean(), 'cv_std': cv_scores_lr.std(),
        'predictions': y_pred_lr
    }
    
    # 2. Random Forest (paramètres conservateurs)
    print("\n2. Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=100,      # Limité pour éviter overfitting
        max_depth=8,           # Profondeur contrôlée
        min_samples_split=20,  # Régularisation
        min_samples_leaf=10,   # Régularisation
        random_state=42
    )
    rf.fit(X_train, y_train)
    
    y_pred_rf = rf.predict(X_test)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    r2_rf = r2_score(y_test, y_pred_rf)
    
    cv_scores_rf = cross_val_score(rf, X_train, y_train, cv=5, scoring='r2')
    
    print(f"   RMSE: ${rmse_rf:.2f}")
    print(f"   R²: {r2_rf:.3f}")
    print(f"   CV R²: {cv_scores_rf.mean():.3f}±{cv_scores_rf.std():.3f}")
    
    models['RandomForest'] = rf
    results['RandomForest'] = {
        'rmse': rmse_rf, 'r2': r2_rf,
        'cv_mean': cv_scores_rf.mean(), 'cv_std': cv_scores_rf.std(),
        'predictions': y_pred_rf
    }
    
    # 3. LightGBM (très conservateur)
    print("\n3. LightGBM...")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=50,       # Très peu d'arbres
        max_depth=6,           # Profondeur limitée
        learning_rate=0.1,     # Learning rate standard
        num_leaves=20,         # Peu de feuilles
        min_child_samples=50,  # Régularisation forte
        subsample=0.8,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    
    y_pred_lgb = lgb_model.predict(X_test)
    rmse_lgb = np.sqrt(mean_squared_error(y_test, y_pred_lgb))
    r2_lgb = r2_score(y_test, y_pred_lgb)
    
    cv_scores_lgb = cross_val_score(lgb_model, X_train, y_train, cv=5, scoring='r2')
    
    print(f"   RMSE: ${rmse_lgb:.2f}")
    print(f"   R²: {r2_lgb:.3f}")
    print(f"   CV R²: {cv_scores_lgb.mean():.3f}±{cv_scores_lgb.std():.3f}")
    
    models['LightGBM'] = lgb_model
    results['LightGBM'] = {
        'rmse': rmse_lgb, 'r2': r2_lgb,
        'cv_mean': cv_scores_lgb.mean(), 'cv_std': cv_scores_lgb.std(),
        'predictions': y_pred_lgb
    }
    
    # Résumé comparatif
    print(f"\n{'='*50}")
    print(f"COMPARAISON MODELES - {city_name.upper()}")
    print(f"{'='*50}")
    
    for model_name, metrics in results.items():
        print(f"{model_name:15} | R²: {metrics['r2']:5.3f} | RMSE: ${metrics['rmse']:6.2f} | CV: {metrics['cv_mean']:5.3f}±{metrics['cv_std']:5.3f}")
    
    # Meilleur modèle
    best_model = max(results.keys(), key=lambda x: results[x]['r2'])
    best_r2 = results[best_model]['r2']
    
    print(f"\nMeilleur modèle: {best_model} (R² = {best_r2:.3f})")
    
    return models, results, scaler, X_test, y_test

def create_evaluation_plots(results_all_cities):
    """
    Crée les graphiques d'évaluation comparatifs
    """
    print(f"\nCREATION GRAPHIQUES EVALUATION")
    print("="*35)
    
    n_cities = len(results_all_cities)
    fig, axes = plt.subplots(1, n_cities, figsize=(6*n_cities, 5))
    
    if n_cities == 1:
        axes = [axes]
    
    for i, (city, (models, results, scaler, X_test, y_test)) in enumerate(results_all_cities.items()):
        ax = axes[i]
        
        # Graphique prédictions vs réel pour le meilleur modèle
        best_model = max(results.keys(), key=lambda x: results[x]['r2'])
        best_predictions = results[best_model]['predictions']
        best_r2 = results[best_model]['r2']
        
        ax.scatter(y_test, best_predictions, alpha=0.6, s=20)
        
        # Ligne parfaite
        min_val, max_val = y_test.min(), y_test.max()
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Prédiction parfaite')
        
        ax.set_xlabel('Prix Réel ($)')
        ax.set_ylabel('Prix Prédit ($)')
        ax.set_title(f'{city} - {best_model}\nR² = {best_r2:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/baseline_evaluation.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Graphiques sauvegardés: {OUTPUT_DIR}/baseline_evaluation.png")

def generate_honest_baseline_report(results_all_cities):
    """
    Génère le rapport du baseline honnête
    """
    print(f"\nGENERATION RAPPORT BASELINE HONNETE")
    print("="*45)
    
    report_lines = []
    report_lines.append("# RAPPORT BASELINE HONNETE - ACTION 3")
    report_lines.append("=" * 50)
    report_lines.append("")
    report_lines.append("## OBJECTIF")
    report_lines.append("Établir une performance de référence légitime après nettoyage des fuites de données.")
    report_lines.append("")
    
    for city, (models, results, _, _, _) in results_all_cities.items():
        report_lines.append(f"## {city.upper()}")
        report_lines.append("")
        
        report_lines.append("| Modèle | RMSE ($) | R² | CV R² |")
        report_lines.append("|--------|----------|----|----|")
        
        for model_name, metrics in results.items():
            report_lines.append(f"| {model_name} | ${metrics['rmse']:.2f} | {metrics['r2']:.3f} | {metrics['cv_mean']:.3f}±{metrics['cv_std']:.3f} |")
        
        # Meilleur modèle
        best_model = max(results.keys(), key=lambda x: results[x]['r2'])
        best_r2 = results[best_model]['r2']
        
        report_lines.append("")
        report_lines.append(f"**Meilleur modèle**: {best_model} (R² = {best_r2:.3f})")
        report_lines.append("")
    
    # Analyse comparative
    report_lines.append("## ANALYSE COMPARATIVE")
    report_lines.append("")
    
    all_r2_scores = []
    for city, (_, results, _, _, _) in results_all_cities.items():
        best_r2 = max(results.values(), key=lambda x: x['r2'])['r2']
        all_r2_scores.append(best_r2)
        report_lines.append(f"- **{city}**: R² = {best_r2:.3f}")
    
    avg_r2 = np.mean(all_r2_scores)
    report_lines.append(f"- **Moyenne**: R² = {avg_r2:.3f}")
    report_lines.append("")
    
    # Évaluation de la performance
    if avg_r2 > 0.3:
        assessment = "Performance satisfaisante pour un modèle honnête"
    elif avg_r2 > 0.15:
        assessment = "Performance modérée mais scientifiquement défendable"
    else:
        assessment = "Performance faible - confirme la complexité intrinsèque de la prédiction"
    
    report_lines.append("## CONCLUSION")
    report_lines.append(f"**{assessment}**")
    report_lines.append("")
    report_lines.append("Ces résultats représentent une performance légitima sans fuites de données,")
    report_lines.append("utilisant uniquement des caractéristiques intrinsèques des propriétés.")
    
    # Sauvegarder
    with open(f"{OUTPUT_DIR}/honest_baseline_report.md", 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Rapport sauvegardé: {OUTPUT_DIR}/honest_baseline_report.md")

def main():
    """
    Action 3 - Modèle baseline honnête principal
    """
    print("ACTION 3 - MODELE BASELINE HONNETE")
    print("="*50)
    
    start_time = pd.Timestamp.now()
    
    try:
        # Charger datasets propres
        datasets = load_clean_datasets()
        
        if not datasets:
            print("Aucun dataset propre disponible")
            return None
        
        results_all_cities = {}
        
        # Traiter chaque ville
        for city_name, df in datasets.items():
            print(f"\n{'='*60}")
            print(f"TRAITEMENT {city_name.upper()}")
            print(f"{'='*60}")
            
            # Préparer données
            X_train, X_test, y_train, y_test = prepare_modeling_data(df, city_name)
            
            # Sauvegarder les données pour les actions suivantes
            baseline_dir = "results/honest_baseline"
            os.makedirs(baseline_dir, exist_ok=True)  # S'assurer que le dossier existe
            X_train.to_csv(f"{baseline_dir}/X_train_{city_name.lower()}.csv", index=False)
            X_test.to_csv(f"{baseline_dir}/X_test_{city_name.lower()}.csv", index=False) 
            pd.DataFrame(y_train, columns=['price']).to_csv(f"{baseline_dir}/y_train_{city_name.lower()}.csv", index=False)
            pd.DataFrame(y_test, columns=['price']).to_csv(f"{baseline_dir}/y_test_{city_name.lower()}.csv", index=False)
            print(f"   Données sauvegardées pour {city_name}")

            if X_train is None:
                print(f"Échec préparation {city_name}")
                continue
            
            # Entraîner modèles
            models, results, scaler, X_test_final, y_test_final = train_baseline_models(
                X_train, X_test, y_train, y_test, city_name
            )
            
            results_all_cities[city_name] = (models, results, scaler, X_test_final, y_test_final)
        
        # Évaluation graphique
        if results_all_cities:
            create_evaluation_plots(results_all_cities)
            generate_honest_baseline_report(results_all_cities)
        
        # Résumé final
        duration = pd.Timestamp.now() - start_time
        
        print(f"\n{'='*60}")
        print("ACTION 3 TERMINEE")
        print(f"{'='*60}")
        print(f"Durée: {duration.total_seconds():.1f} secondes")
        
        for city, (_, results, _, _, _) in results_all_cities.items():
            best_r2 = max(results.values(), key=lambda x: x['r2'])['r2']
            print(f"{city}: R² = {best_r2:.3f}")
        
        print(f"\nResultats honnêtes obtenus - Prêt pour Action 4")
        
        return results_all_cities
        
    except Exception as e:
        print(f"\n❌ Erreur Action 3: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = main()