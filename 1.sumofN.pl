% Base case: The sum of integers from 1 to 0 is 0
sum(0, 0).

% Recursive case: Sum from 1 to N is N + sum from 1 to N-1
sum(N, Sum) :-
    N > 0, % Ensure N is positive
    N1 is N - 1, % Decrement N by 1
    sum(N1, Sum1), % Recursive call
    Sum is N + Sum1. % Add the current N to the sum
