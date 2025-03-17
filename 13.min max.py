# Minimax function to calculate the optimal move
def minimax(board, depth, is_maximizing):
    # Base case: Check for winner or draw
    score = evaluate_board(board)
    if score == 10 or score == -10:
        return score
    if is_draw(board):
        return 0

    # Maximizing player's turn
    if is_maximizing:
        best_score = -float('inf')
        for move in get_available_moves(board):
            board[move] = 'X'  # Assuming 'X' is the maximizing player
            best_score = max(best_score, minimax(board, depth + 1, False))
            board[move] = ' '  # Undo move
        return best_score

    # Minimizing player's turn
    else:
        best_score = float('inf')
        for move in get_available_moves(board):
            board[move] = 'O'  # Assuming 'O' is the minimizing player
            best_score = min(best_score, minimax(board, depth + 1, True))
            board[move] = ' '  # Undo move
        return best_score

# Function to evaluate the board and return a score
def evaluate_board(board):
    winning_combinations = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
                            (0, 3, 6), (1, 4, 7), (2, 5, 8),
                            (0, 4, 8), (2, 4, 6)]
    for combo in winning_combinations:
        if board[combo[0]] == board[combo[1]] == board[combo[2]] == 'X':
            return 10
        if board[combo[0]] == board[combo[1]] == board[combo[2]] == 'O':
            return -10
    return 0

# Check if the game is a draw
def is_draw(board):
    return ' ' not in board

# Get all available moves on the board
def get_available_moves(board):
    return [i for i, spot in enumerate(board) if spot == ' ']

# Example usage
if __name__ == "__main__":
    # Example board configuration (Tic-Tac-Toe):
    board = ['X', 'O', 'X',
             'O', 'X', ' ',
             ' ', 'O', ' ']

    # Determine the best move for the maximizing player
    best_move = None
    best_value = -float('inf')
    for move in get_available_moves(board):
        board[move] = 'X'
        move_value = minimax(board, 0, False)
        board[move] = ' '  # Undo move

        if move_value > best_value:
            best_value = move_value
            best_move = move

    print(f"The best move for 'X' is at position {best_move}")
