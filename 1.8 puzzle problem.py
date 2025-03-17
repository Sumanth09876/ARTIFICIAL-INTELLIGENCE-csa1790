import numpy as np
from queue import PriorityQueue

class PuzzleState:
    def __init__(self, board, moves=0):
        self.board = board
        self.blank_tile = self.find_blank_tile()
        self.moves = moves
        self.priority = self.moves + self.heuristic()

    def find_blank_tile(self):
        return tuple(np.argwhere(self.board == 0)[0])

    def heuristic(self):
        distance = 0
        for i in range(3):
            for j in range(3):
                if self.board[i][j] != 0:
                    target_x = (self.board[i][j] - 1) // 3
                    target_y = (self.board[i][j] - 1) % 3
                    distance += abs(target_x - i) + abs(target_y - j)
        return distance

    def get_neighbors(self):
        neighbors = []
        x, y = self.blank_tile
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < 3 and 0 <= new_y < 3:
                new_board = self.board.copy()
                new_board[x][y], new_board[new_x][new_y] = new_board[new_x][new_y], new_board[x][y]
                neighbors.append(PuzzleState(new_board, self.moves + 1))
        return neighbors

    def __lt__(self, other):
        return self.priority < other.priority

def a_star(start_board):
    start_state = PuzzleState(start_board)
    goal_state = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
    visited = set()
    queue = PriorityQueue()
    queue.put(start_state)

    while not queue.empty():
        current_state = queue.get()
        if np.array_equal(current_state.board, goal_state):
            return current_state.moves

        visited.add(tuple(map(tuple, current_state.board)))
        for neighbor in current_state.get_neighbors():
            if tuple(map(tuple, neighbor.board)) not in visited:
                queue.put(neighbor)

    return -1

if __name__ == "__main__":
    initial_board = np.array([[1, 2, 3], [4, 0, 5], [7, 8, 6]])
    moves = a_star(initial_board)
    print(f"Minimum moves to solve the puzzle: {moves}")
