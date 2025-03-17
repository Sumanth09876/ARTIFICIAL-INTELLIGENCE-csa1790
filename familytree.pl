% Facts about family members (parent-child relationships)
parent(john, mary).     % John is Mary's parent
parent(john, mike).     % John is Mike's parent
parent(mary, alice).    % Mary is Alice's parent
parent(mary, bob).      % Mary is Bob's parent
parent(mike, charlie).  % Mike is Charlie's parent
parent(mike, sophie).   % Mike is Sophie's parent
parent(alice, emily).   % Alice is Emily's parent

% Defining sibling relationships: Two people are siblings if they share a common parent.
sibling(X, Y) :-
    parent(P, X),
    parent(P, Y),
    X \= Y.  % X and Y should not be the same person

% Defining grandparent relationship: A person is a grandparent if they have a child who is a parent.
grandparent(X, Y) :-
    parent(X, Z),
    parent(Z, Y).

% Defining ancestor relationship: A person is an ancestor if they are a parent or grandparent (or more) of another person.
ancestor(X, Y) :-
    parent(X, Y).  % Direct parent
ancestor(X, Y) :-
    parent(X, Z),
    ancestor(Z, Y).  % Grandparent, great-grandparent, etc.

% Query to find if two people are siblings
% Example: sibling(mary, mike) would check if Mary and Mike are siblings.

% Query to find the grandparent of a person
% Example: grandparent(john, alice) would check if John is Alice's grandparent.

% Query to find the ancestors of a person
% Example: ancestor(john, sophie) would check if John is an ancestor of Sophie.
