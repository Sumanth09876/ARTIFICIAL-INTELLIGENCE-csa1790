% Facts: Symptoms
symptom(fever).
symptom(cough).
symptom(fatigue).
symptom(headache).

% Facts: Diseases and their symptoms
disease(covid19, [fever, cough, fatigue]).
disease(flu, [fever, cough, headache]).
disease(cold, [cough, headache]).
disease(malaria, [fever, fatigue]).

% Backward Chaining for diagnosis: Starts with a goal (disease)
diagnose(Disease) :-
    disease(Disease, Symptoms),
    check_symptoms(Symptoms).

% Rule to check if all symptoms of the disease are present
check_symptoms([]).
check_symptoms([Symptom | Rest]) :-
    symptom(Symptom),
    check_symptoms(Rest).

% Rule to check if the symptom is present
has_symptom(Symptom) :-
    symptom(Symptom).

% Query to diagnose a disease based on the symptoms
start_diagnosis :-
    diagnose(covid19),
    write('Diagnosis: covid19'), nl.
