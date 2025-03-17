% Facts representing diseases and the associated diet restrictions
diet_for(diabetes, 'Low sugar, Low carbohydrate diet').
diet_for(hypertension, 'Low sodium, Low salt diet').
diet_for(obesity, 'Low calorie, High protein diet').
diet_for(arthritis, 'Anti-inflammatory diet, High omega-3 fatty acids').
diet_for(heart_disease, 'Low fat, Low cholesterol diet').
diet_for(celiac, 'Gluten-free diet').
diet_for(allergy, 'Avoid allergens, High fiber diet').

% Rules to recommend diets based on disease
recommend_diet(Disease, Diet) :-
    diet_for(Disease, Diet).

% Query to check which diet is recommended for a particular disease
% Example: recommend_diet(diabetes, Diet).
