from collections import deque
def missionaries_cannibals():
    initial_state = (3, 3, True) 
    goal_state = (0, 0, False)
    visited = set() 
    queue = deque() 
    queue.append((initial_state, []))
    visited.add(initial_state)
    while queue:
        current_state, path = queue.popleft()
        missionaries_left, cannibals_left, boat_position = current_state
        if current_state == goal_state:
            return path
        next_states = generate_next_states(missionaries_left, cannibals_left, boat_position)
        for next_state in next_states:
            if next_state not in visited and is_valid_state(next_state):
                visited.add(next_state)
                queue.append((next_state, path + [next_state]))
    return "No solution found"
def generate_next_states(m, c, boat_position):
    next_states = []
    if boat_position:
        possible_moves = [
            (m - 2, c, False),
            (m - 1, c, False),
            (m, c - 2, False),
            (m, c - 1, False),
            (m - 1, c - 1, False) 
        ]
    else:
        possible_moves = [
            (m + 2, c, True),
            (m + 1, c, True),
            (m, c + 2, True), 
            (m, c + 1, True), 
            (m + 1, c + 1, True)
        ]
    for move in possible_moves:
        if is_valid_state(move):
            next_states.append(move)
    return next_states
def is_valid_state(state):
    m, c, boat_position = state
    if m < 0 or c < 0 or m > 3 or c > 3:
        return False
    if m < c and m > 0:
        return False
    if (3 - m) < (3 - c) and (3 - m) > 0:
        return False
    return True
solution = missionaries_cannibals()
if isinstance(solution, list):
    print("Solution found:")
    for step in solution:
        print(step)
else:
    print(solution)
