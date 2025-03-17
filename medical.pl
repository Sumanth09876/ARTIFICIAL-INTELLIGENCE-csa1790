% Facts representing symptoms
symptom(fever).
symptom(cough).
symptom(fatigue).
symptom(headache).
symptom(sore_throat).
symptom(nausea).
symptom(shortness_of_breath).
symptom(body_ache).

% Facts representing diseases and their associated symptoms
disease(covid19, [fever, cough, fatigue, body_ache, shortness_of_breath]).
disease(flu, [fever, cough, sore_throat, body_ache, fatigue]).
disease(cold, [cough, sore_throat, body_ache]).
disease(malaria, [fever, fatigue, body_ache, nausea]).
disease(allergy, [sore_throat, body_ache, fatigue]).

% Rule to diagnose a disease based on the symptoms provided
diagnose(Disease) :-
    disease(Disease, Symptoms),
    check_symptoms(Symptoms).

% Helper rule to check if the symptoms match the given disease
check_symptoms([]).
check_symptoms([Symptom|Rest]) :-
    symptom(Symptom),
    check_symptoms(Rest).

% Rule to check if a symptom is present in the list of symptoms
has_symptom(Symptom) :-
    symptom(Symptom).
