# RAPPORT RECONSTRUCTION DATASETS PROPRES
==================================================

## PARIS

- **Colonnes originales**: 258
- **Colonnes finales**: 31
- **Réduction**: 227 colonnes éliminées (88.0%)
- **Échantillons**: 53,455

### Statistiques Prix
- Médiane: $160
- Moyenne: $233
- Écart-type: $226

## SEATTLE

- **Colonnes originales**: 167
- **Colonnes finales**: 31
- **Réduction**: 136 colonnes éliminées (81.4%)
- **Échantillons**: 3,818

### Statistiques Prix
- Médiane: $100
- Moyenne: $128
- Écart-type: $90

## FEATURES CONSERVEES

 1. accommodates
 2. bedrooms
 3. beds
 4. bathrooms
 5. latitude
 6. longitude
 7. neighbourhood_cleansed
 8. property_type
 9. room_type
10. minimum_nights
11. availability_30
12. availability_365
13. number_of_reviews
14. reviews_per_month
15. review_scores_rating
16. review_scores_accuracy
17. review_scores_cleanliness
18. review_scores_checkin
19. review_scores_communication
20. review_scores_location
21. review_scores_value
22. month
23. day_of_week
24. nb_amenities
25. amenities
26. distance_center_km
27. lat_from_median
28. lon_from_median
29. people_per_bedroom
30. amenities_per_person
31. estimated_months_active

## VALIDATION
- ✅ Aucune fuite de données détectée
- ✅ Corrélations avec prix < 0.7
- ✅ Features exclusivement intrinsèques
- ✅ Prêt pour modélisation légitime