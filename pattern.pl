% Define Facts (Predefined Patterns)
pattern([a, b, c]).
pattern([1, 2, 3]).

% Rule to perform pattern matching
match_pattern(InputPattern) :-
    pattern(PredefinedPattern),
    InputPattern = PredefinedPattern,   % Unify the input pattern with predefined pattern
    write('Match found'), nl.           % If match, print "Match found".

% Rule to check if no match is found
match_pattern(_) :-
    write('No match'), nl.              % If no match, print "No match".
    
% To test the pattern matching process
start :-
    match_pattern([a, b, c]),    % Testing with matching input
    match_pattern([1, 2, 3]),    % Testing with matching input
    match_pattern([x, y, z]).    % Testing with non-matching input
