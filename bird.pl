% Facts representing birds and their ability to fly
bird(canary).
bird(eagle).
bird(penguin).

can_fly(canary).
can_fly(eagle).

% Rule to check if a bird can fly
can_fly(Bird) :-
    bird(Bird),
    can_fly(Bird).

% Rule to check if a bird cannot fly
cannot_fly(Bird) :-
    bird(Bird),
    \+ can_fly(Bird).
