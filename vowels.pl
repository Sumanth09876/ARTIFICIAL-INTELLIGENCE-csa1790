% Define vowel characters
vowel(a).
vowel(e).
vowel(i).
vowel(o).
vowel(u).

% Base case: If the list is empty, there are no vowels
count_vowels([], 0).

% Recursive case: If the head of the list is a vowel, increment the count
count_vowels([H|T], Count) :-
    vowel(H),  % Check if the head is a vowel
    count_vowels(T, CountTail),  % Recursively count the vowels in the tail
    Count is CountTail + 1.  % Increment the count if the head is a vowel

% Recursive case: If the head of the list is not a vowel, just continue with the tail
count_vowels([H|T], Count) :-
    \+ vowel(H),  % Check if the head is NOT a vowel
    count_vowels(T, Count).  % Continue recursively with the tail
