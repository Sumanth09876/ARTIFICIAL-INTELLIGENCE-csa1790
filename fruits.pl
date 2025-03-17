% Facts about fruits and colors
fruit(apple).
fruit(banana).
fruit(grape).

color(red).
color(yellow).
color(purple).

% We want to assign a color to each fruit
assign_colors([apple, banana, grape], [red, yellow, purple]).

% Rules to assign colors to fruits
assign_colors([Fruit|RestFruits], [Color|RestColors]) :-
    fruit(Fruit),
    color(Color),
    assign_colors(RestFruits, RestColors).

% Ensure each fruit gets a unique color
assign_unique_colors([], []).
assign_unique_colors([Fruit|RestFruits], [Color|RestColors]) :-
    fruit(Fruit),
    color(Color),
    not_member(Color, RestColors),
    assign_unique_colors(RestFruits, RestColors).

% Ensure a color is not already used
not_member(_, []).
not_member(X, [Y|Rest]) :-
    X \= Y,
    not_member(X, Rest).

% Query to find valid fruit-color assignments
find_fruit_color_assignments(Fruits, Colors) :-
    assign_unique_colors(Fruits, Colors).
