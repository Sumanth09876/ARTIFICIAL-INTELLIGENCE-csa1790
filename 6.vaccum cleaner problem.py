import numpy as np

def vacuum_cleaner(grid, start_x, start_y):
    rows, cols = len(grid), len(grid[0])
    total_moves = 0
    
    for i in range(rows):
        for j in range(cols):
            x, y = (i, j) if i % 2 == 0 else (i, cols - 1 - j)  # Zigzag movement
            if grid[x][y] == 1:
                print(f"Cleaning dirt at ({x}, {y})")
                grid[x][y] = 0
            total_moves += 1
    
    return grid, total_moves

# Input: Initial grid and vacuum position
grid = [
    [1, 0], 
    [1, 1]
]
start_x, start_y = 0, 0  # Vacuum starts at (0,0)

# Process
cleaned_grid, moves = vacuum_cleaner(grid, start_x, start_y)

# Output
print("\nFinal Cleaned Grid:")
print(np.array(cleaned_grid))
print(f"Total Moves: {moves}")
