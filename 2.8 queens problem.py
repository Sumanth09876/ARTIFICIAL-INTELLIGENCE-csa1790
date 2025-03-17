# 8 Queens Problem

def print_board(board):
    for row in board:
        print(" ".join(row))
    print("\n")

def is_safe(board, row, col):
    for i in range(col):
        if board[row][i] == "Q":
            return False
    for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
        if board[i][j] == "Q":
            return False
    for i, j in zip(range(row, len(board)), range(col, -1, -1)):
        if board[i][j] == "Q":
            return False
    return True

def solve_n_queens_util(board, col):
    if col >= len(board):
        print_board(board)
        return True
    res = False
    for i in range(len(board)):
        if is_safe(board, i, col):
            board[i][col] = "Q"
            res = solve_n_queens_util(board, col + 1) or res
            board[i][col] = "."
    return res

def solve_n_queens(n):
    board = [["." for _ in range(n)] for _ in range(n)]
    if not solve_n_queens_util(board, 0):
        print("Solution does not exist")
    return

solve_n_queens(8)
