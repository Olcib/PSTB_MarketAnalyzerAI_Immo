# RAPPORT BASELINE HONNETE - ACTION 3
==================================================

## OBJECTIF
Établir une performance de référence légitime après nettoyage des fuites de données.

## PARIS

| Modèle | RMSE ($) | R² | CV R² |
|--------|----------|----|----|
| LinearRegression | $174.48 | 0.386 | 0.380±0.025 |
| RandomForest | $166.68 | 0.440 | 0.442±0.004 |
| LightGBM | $164.39 | 0.455 | 0.454±0.004 |

**Meilleur modèle**: LightGBM (R² = 0.455)

## SEATTLE

| Modèle | RMSE ($) | R² | CV R² |
|--------|----------|----|----|
| LinearRegression | $62.75 | 0.558 | 0.525±0.055 |
| RandomForest | $60.63 | 0.587 | 0.547±0.041 |
| LightGBM | $61.25 | 0.579 | 0.550±0.042 |

**Meilleur modèle**: RandomForest (R² = 0.587)

## ANALYSE COMPARATIVE

- **Paris**: R² = 0.455
- **Seattle**: R² = 0.587
- **Moyenne**: R² = 0.521

## CONCLUSION
**Performance satisfaisante pour un modèle honnête**

Ces résultats représentent une performance légitima sans fuites de données,
utilisant uniquement des caractéristiques intrinsèques des propriétés.