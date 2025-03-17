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

% Rule to apply forward chaining
diagnose(Disease) :-
    disease(Disease, Symptoms),
    can_infer(Symptoms),
    write('Diagnosis: '), write(Disease), nl.

% Rule to infer whether all symptoms are present
can_infer([]).
can_infer([Symptom | Rest]) :-
    symptom(Symptom),
    can_infer(Rest).

% Adding new facts (simulating user input or symptom detection)
add_symptom(Symptom) :-
    \+ symptom(Symptom),
    assert(symptom(Symptom)),
    write('Symptom added: '), write(Symptom), nl.

% For testing purposes: Let's assume the symptoms are added dynamically
start_diagnosis :-
    add_symptom(fever),
    add_symptom(cough),
    add_symptom(fatigue),
    diagnose(Disease).
