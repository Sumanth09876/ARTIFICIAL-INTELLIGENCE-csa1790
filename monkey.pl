% Facts representing the initial state
% monkey_at(Height) represents the current height of the monkey
% box_at(Height) represents the height of the box
% bananas_at(Height) represents the height at which the bananas are hanging
% monkey_has_banana is true if the monkey has the banana, false otherwise

% Initial state: Monkey is at ground level, box is at ground level, bananas are 3 meters high.
monkey_at(0).
box_at(0).
bananas_at(3).
monkey_has_banana(false).

% The monkey can move to a new height if there's no obstruction
move_up(Height, NewHeight) :-
    monkey_at(CurrentHeight),
    NewHeight is CurrentHeight + Height,
    NewHeight =< 3.

% The monkey can push the box to a new height
push_box(Height, NewHeight) :-
    box_at(CurrentHeight),
    NewHeight is CurrentHeight + Height,
    NewHeight =< 3.

% The monkey can grab the banana if it reaches the height of the bananas
grab_banana :-
    monkey_at(Height),
    bananas_at(Height),
    \+ monkey_has_banana,
    assert(monkey_has_banana(true)).

% If the monkey is below the bananas, it can move or push the box
plan(Action) :-
    (Action = move_up(1), move_up(1), write('Monkey moves up 1 meter.'));
    (Action = push_box(1), push_box(1), write('Monkey pushes the box up by 1 meter.'));
    (Action = grab_banana, grab_banana, write('Monkey grabs the bananas.')).

% Check if the monkey has the bananas
goal :-
    monkey_has_banana(true),
    write('Monkey has the banana!').
