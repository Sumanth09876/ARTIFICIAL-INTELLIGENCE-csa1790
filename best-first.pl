% Facts: Each city is connected to another city with a distance.
% travel(City1, City2, Cost) represents the distance between City1 and City2
travel(london, paris, 2).
travel(london, berlin, 3).
travel(paris, berlin, 1).
travel(paris, rome, 4).
travel(berlin, rome, 2).
travel(rome, madrid, 5).

% Heuristic function: estimated cost (straight-line distance to the goal).
% heuristic(City, HeuristicCost)
heuristic(london, 5).
heuristic(paris, 3).
heuristic(berlin, 2).
heuristic(rome, 1).
heuristic(madrid, 0). % Goal city

% Best-First Search (main function)
best_first_search(Start, Goal, Path) :-
    best_first([Start], Goal, [], Path).

% Base case: if the start is the goal, we have found the path
best_first([Goal|Path], Goal, Path, [Goal|Path]).

% Recursive case: Explore neighbors and continue search
best_first([Current|Rest], Goal, Explored, Path) :-
    findall(Next, (travel(Current, Next, _), \+ member(Next, Explored)), Neighbors),
    add_heuristic(Neighbors, Goal, Current, NewNeighbors),
    append(Rest, NewNeighbors, Frontier),
    best_first(Frontier, Goal, [Current|Explored], Path).

% Add heuristic cost to the neighbors list to guide the search
add_heuristic([], _, _, []).
add_heuristic([City|Rest], Goal, Current, [(City, FScore)|RestWithFScore]) :-
    heuristic(City, H),
    travel(Current, City, StepCost),
    FScore is StepCost + H,
    add_heuristic(Rest, Goal, Current, RestWithFScore).
