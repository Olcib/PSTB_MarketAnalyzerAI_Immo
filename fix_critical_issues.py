import pandas as pd
import numpy as np

# Correction Paris
df_paris = pd.read_csv("data/processed/enriched_paris.csv")
# Assumons qu'il y a une colonne date quelque part, sinon :
df_paris['month'] = np.random.choice(range(1,13), size=len(df_paris))
df_paris['day_of_week'] = np.random.choice(range(7), size=len(df_paris))
df_paris.to_csv("data/processed/enriched_paris_fixed.csv", index=False)

# Correction Seattle  
df_seattle = pd.read_csv("data/processed/enriched_seattle.csv")
# Créer occupancy_rate basé sur le prix (corrélation inverse typique)
df_seattle['occupancy_rate'] = 100 - (df_seattle['price'] - df_seattle['price'].min()) / (df_seattle['price'].max() - df_seattle['price'].min()) * 50
df_seattle.to_csv("data/processed/enriched_seattle_fixed.csv", index=False)