% Base case: If there is only one disk, move it from source to destination
hanoi(1, Source, Destination, _) :-
    write('Move disk from '), write(Source), write(' to '), write(Destination), nl.

% Recursive case: Move N disks from Source to Destination using Aux as auxiliary
hanoi(N, Source, Destination, Aux) :-
    N > 1,
    M is N - 1,
    hanoi(M, Source, Aux, Destination),   % Move N-1 disks from Source to Aux using Destination
    hanoi(1, Source, Destination, _),     % Move the last disk from Source to Destination
    hanoi(M, Aux, Destination, Source).   % Move N-1 disks from Aux to Destination using Source
