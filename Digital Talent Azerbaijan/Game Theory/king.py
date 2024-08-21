def find_game_value(board):
    n = len(board)
    m = len(board[0])

    # Create a dp table to store the maximum value at each cell
    dp = [[0] * m for _ in range(n)]

    # Start from the bottom-left cell
    dp[n-1][0] = board[n-1][0]

    # Iterate over each cell in the board
    for i in range(n-2, -1, -1):
        dp[i][0] = dp[i+1][0] + board[i][0]

    for j in range(1, m):
        dp[n-1][j] = dp[n-1][j-1] + board[n-1][j]

    for i in range(n-2, -1, -1):
        for j in range(1, m):
            dp[i][j] = max(dp[i+1][j], dp[i][j-1]) + board[i][j]

    return dp[0][m-1]

# Sample input
board = [
    [0, 1, 0, 1, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0]
]

# Find the value of the game
value = find_game_value(board)
print(value)