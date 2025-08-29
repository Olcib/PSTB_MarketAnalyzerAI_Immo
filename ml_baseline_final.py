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
    Charge et prépare les données avec nettoyage robuste
    """
    print("🔍 CHARGEMENT ET PRÉPARATION DES DONNÉES")
    print("="*70)
    
    # Chargement
    df_paris = pd.read_csv(PARIS_FILE)
    df_seattle = pd.read_csv(SEATTLE_FILE)
    
    print(f"✅ Paris : {df_paris.shape[0]:,} lignes × {df_paris.shape[1]} colonnes")
    print(f"✅ Seattle : {df_seattle.shape[0]:,} lignes × {df_seattle.shape[1]} colonnes")
    
    # Préparation des datasets avec nettoyage
    def prepare_city_data(df, city_name):
        print(f"\n🏙️  Préparation {city_name}...")
        
        df_clean = df.copy()
        df_clean['city'] = city_name
        
        # Mapping des colonnes essentielles
        column_mapping = {}
        
        # Occupancy
        if city_name == 'Paris' and 'estimated_occupancy_l365d' in df_clean.columns:
            column_mapping['estimated_occupancy_l365d'] = 'occupancy_rate'
            print("  📊 Occupancy: estimated_occupancy_l365d → occupancy_rate")
        elif 'occupancy_rate' in df_clean.columns:
            print("  📊 Occupancy: occupancy_rate ✅")
        
        # Revenue
        if 'price' in df_clean.columns:
            column_mapping['price'] = 'revenue'
            print("  💰 Revenue: price → revenue")
        
        # Sentiment
        if 'review_scores_rating' in df_clean.columns:
            column_mapping['review_scores_rating'] = 'sentiment_score'
            print("  ⭐ Sentiment: review_scores_rating → sentiment_score")
        
        # Amenities
        if 'nb_amenities' in df_clean.columns:
            print("  🏠 Amenities: nb_amenities ✅")
        
        # Appliquer le mapping
        df_clean = df_clean.rename(columns=column_mapping)
        
        # Nettoyage des colonnes temporelles
        if 'day_of_week' in df_clean.columns:
            # Convertir tout en string puis mapper
            df_clean['day_of_week'] = df_clean['day_of_week'].astype(str)
            
            # Mapping des jours
            day_mapping = {
                'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
                'Friday': 4, 'Saturday': 5, 'Sunday': 6,
                '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
                'nan': 0  # valeur par défaut
            }
            
            df_clean['day_of_week'] = df_clean['day_of_week'].map(day_mapping).fillna(0).astype(int)
            print("  📅 day_of_week nettoyé et converti")
        
        if 'month' in df_clean.columns:
            df_clean['month'] = pd.to_numeric(df_clean['month'], errors='coerce').fillna(1).astype(int)
            print("  📅 month nettoyé et converti")
        
        return df_clean
    
    # Préparation des deux villes
    df_paris_clean = prepare_city_data(df_paris, 'Paris')
    df_seattle_clean = prepare_city_data(df_seattle, 'Seattle')
    
    # Sélection des colonnes finales
    target_cols = ['occupancy_rate', 'revenue', 'sentiment_score', 'nb_amenities', 'month', 'day_of_week', 'city']
    
    # Sélection pour Paris
    paris_cols = [col for col in target_cols if col in df_paris_clean.columns]
    df_paris_final = df_paris_clean[paris_cols].copy()
    
    # Sélection pour Seattle
    seattle_cols = [col for col in target_cols if col in df_seattle_clean.columns]
    df_seattle_final = df_seattle_clean[seattle_cols].copy()
    
    print(f"\n✅ Paris final: {len(paris_cols)-1}/6 features")
    print(f"✅ Seattle final: {len(seattle_cols)-1}/6 features")
    
    # Combinaison
    df_combined = pd.concat([df_paris_final, df_seattle_final], ignore_index=True)
    print(f"✅ Dataset combiné: {df_combined.shape[0]:,} lignes × {df_combined.shape[1]} colonnes")
    
    # Résumé des features
    print(f"\n📋 RÉSUMÉ DES FEATURES:")
    for col in df_combined.columns:
        if col != 'city':
            missing_count = int(df_combined[col].isnull().sum())
            total_count = len(df_combined)
            missing_pct = (missing_count / total_count) * 100
            unique_count = int(df_combined[col].nunique())
            print(f"  • {col}: {unique_count} valeurs uniques, {missing_pct:.1f}% manquantes")
    
    return df_combined

def prepare_targets_and_features(df_combined):
    """
    Prépare les variables cibles et features
    """
    print(f"\n🎯 PRÉPARATION DES VARIABLES CIBLES ET FEATURES")
    print("="*70)
    
    # Variables cibles
    print("🎯 Variables cibles...")
    
    # Régression
    if 'revenue' in df_combined.columns:
        y_regression = df_combined['revenue'].copy()
        print(f"  ✅ Régression: revenue")
        print(f"    • Min: ${float(y_regression.min()):.2f}")
        print(f"    • Max: ${float(y_regression.max()):.2f}")
        print(f"    • Moyenne: ${float(y_regression.mean()):.2f}")
        print(f"    • Manquantes: {int(y_regression.isnull().sum())}")
    else:
        y_regression = None
        print("  ❌ Pas de variable revenue")
    
    # Classification
    if 'sentiment_score' in df_combined.columns:
        sentiment_col = df_combined['sentiment_score']
        max_score = float(sentiment_col.max())
        
        if max_score <= 5:
            threshold = 4.5
        elif max_score <= 100:
            threshold = 80
        else:
            threshold = float(sentiment_col.median())
        
        y_classification = (sentiment_col >= threshold).astype(int)
        
        class_0 = int((y_classification == 0).sum())
        class_1 = int((y_classification == 1).sum())
        total = len(y_classification)
        
        print(f"  ✅ Classification: sentiment >= {threshold}")
        print(f"    • Mal noté (0): {class_0:,} ({class_0/total*100:.1f}%)")
        print(f"    • Bien noté (1): {class_1:,} ({class_1/total*100:.1f}%)")
    else:
        y_classification = None
        print("  ❌ Pas de variable sentiment_score")
    
    # Préparation des features
    print(f"\n🔧 Préparation des features...")
    
    # Exclure les targets
    exclude_cols = ['revenue', 'sentiment_score']
    feature_cols = [col for col in df_combined.columns if col not in exclude_cols]
    X = df_combined[feature_cols].copy()
    
    print(f"  📊 Features initiales: {list(X.columns)}")
    
    # Encodage de la variable city
    if 'city' in X.columns:
        le_city = LabelEncoder()
        X['city_encoded'] = le_city.fit_transform(X['city'].fillna('Unknown'))
        X = X.drop('city', axis=1)
        print(f"  ✅ City encodé: {int(X['city_encoded'].nunique())} valeurs")
    
    # Vérifier que toutes les colonnes sont numériques
    print(f"\n🔍 Vérification des types de données:")
    for col in X.columns:
        dtype = X[col].dtype
        if dtype == 'object':
            print(f"  ⚠️  {col}: type object - conversion forcée")
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        print(f"  ✅ {col}: {X[col].dtype}")
    
    # Gestion des valeurs manquantes
    missing_count = int(X.isnull().sum().sum())
    if missing_count > 0:
        print(f"\n📊 {missing_count} valeurs manquantes détectées - imputation...")
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(
            imputer.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
    else:
        print(f"\n📊 Aucune valeur manquante")
        X_imputed = X.copy()
    
    # Normalisation
    print(f"🔧 Normalisation...")
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_imputed),
        columns=X_imputed.columns,
        index=X_imputed.index
    )
    
    print(f"✅ Features finales prêtes: {list(X_scaled.columns)}")
    
    return X_scaled, y_regression, y_classification

def train_models(X, y_reg, y_clf):
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
        
        # Nettoyage
        mask_reg = ~y_reg.isnull()
        X_reg = X[mask_reg]
        y_reg_clean = y_reg[mask_reg]
        
        print(f"  📊 Données: {len(X_reg):,} échantillons")
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_reg, y_reg_clean, test_size=0.2, random_state=42
        )
        
        print(f"  📊 Train: {len(X_train):,} | Test: {len(X_test):,}")
        
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
        
        # Nettoyage
        mask_clf = ~y_clf.isnull()
        X_clf = X[mask_clf]
        y_clf_clean = y_clf[mask_clf]
        
        print(f"  📊 Données: {len(X_clf):,} échantillons")
        
        if y_clf_clean.nunique() > 1:
            # Split
            X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
                X_clf, y_clf_clean, test_size=0.2, random_state=42, stratify=y_clf_clean
            )
            
            print(f"  📊 Train: {len(X_train_clf):,} | Test: {len(X_test_clf):,}")
            
            # Random Forest
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
            print(f"  ⚠️  Une seule classe - classification impossible")
    
    return results

def create_visualizations(results):
    """
    Crée les graphiques
    """
    if not results:
        print("⚠️  Pas de résultats à visualiser")
        return
    
    print(f"\n📊 CRÉATION DES GRAPHIQUES")
    print("-" * 30)
    
    n_plots = len(results)
    if n_plots == 0:
        return
    
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # Régression
    if 'regression' in results:
        ax = axes[plot_idx]
        X_test, y_test, y_pred_lr, y_pred_rf = results['regression']['test_data']
        
        lr_r2 = results['regression']['LinearRegression']['r2']
        rf_r2 = results['regression']['RandomForest']['r2']
        
        ax.scatter(y_test, y_pred_lr, alpha=0.6, label=f"Linear (R²={lr_r2:.3f})", s=20)
        ax.scatter(y_test, y_pred_rf, alpha=0.6, label=f"RF (R²={rf_r2:.3f})", s=20)
        
        min_val, max_val = y_test.min(), y_test.max()
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Parfait')
        
        ax.set_xlabel('Prix Réel ($)')
        ax.set_ylabel('Prix Prédit ($)')
        ax.set_title('Régression: Prédictions vs Réel')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plot_idx += 1
    
    # Classification
    if 'classification' in results:
        ax = axes[plot_idx]
        cm = results['classification']['confusion_matrix']
        accuracy = results['classification']['accuracy']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Mal noté', 'Bien noté'],
                   yticklabels=['Mal noté', 'Bien noté'])
        ax.set_title(f"Classification (Acc: {accuracy:.3f})")
        ax.set_xlabel('Prédiction')
        ax.set_ylabel('Réel')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plots/ml_baseline_results.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Graphiques: {OUTPUT_DIR}/plots/ml_baseline_results.png")

def generate_report(results):
    """
    Génère le rapport final
    """
    print(f"\n📋 RAPPORT FINAL")
    print("-" * 20)
    
    # Sauvegarde rapport
    report_content = "# RAPPORT ML BASELINE - JOUR 3\n\n"
    
    if 'regression' in results:
        lr_r2 = results['regression']['LinearRegression']['r2']
        rf_r2 = results['regression']['RandomForest']['r2']
        report_content += f"## RÉGRESSION\n"
        report_content += f"- Linear Regression R²: {lr_r2:.3f}\n"
        report_content += f"- Random Forest R²: {rf_r2:.3f}\n\n"
    
    if 'classification' in results:
        accuracy = results['classification']['accuracy']
        report_content += f"## CLASSIFICATION\n"
        report_content += f"- Accuracy: {accuracy:.3f}\n\n"
    
    with open(f"{OUTPUT_DIR}/ml_baseline_report.md", 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"✅ Rapport: {OUTPUT_DIR}/ml_baseline_report.md")
    
    return results

def main():
    """
    Fonction principale finale
    """
    print("🚀 JOUR 3 - ML BASELINE (VERSION FINALE)")
    print("="*70)
    
    try:
        # 1. Chargement et préparation
        df_combined = load_and_prepare_data()
        
        # 2. Variables cibles et features
        X, y_regression, y_classification = prepare_targets_and_features(df_combined)
        
        # 3. Entraînement
        results = train_models(X, y_regression, y_classification)
        
        # 4. Visualisations
        create_visualizations(results)
        
        # 5. Rapport
        generate_report(results)
        
        # 6. Résumé final
        print(f"\n🏆 RÉSULTATS FINAUX")
        print("="*40)
        
        if 'regression' in results:
            lr_r2 = results['regression']['LinearRegression']['r2']
            rf_r2 = results['regression']['RandomForest']['r2']
            best_r2 = max(lr_r2, rf_r2)
            best_model = 'Random Forest' if rf_r2 > lr_r2 else 'Linear Regression'
            print(f"📊 RÉGRESSION: {best_model} - R² = {best_r2:.3f}")
        
        if 'classification' in results:
            accuracy = results['classification']['accuracy']
            print(f"🎯 CLASSIFICATION: Accuracy = {accuracy:.3f}")
        
        print("="*40)
        print(f"✅ ML BASELINE TERMINÉ !")
        print(f"📁 Résultats: {OUTPUT_DIR}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Erreur: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()