def find_winner(A, B, A_moves, B_moves):
    # Check if the first player can force a win
    for move in A_moves:
        if A - move <= 0 and B <= 0:
            return "First"
        if A - move >= 0 and B <= 0:
            return "First"
        if A <= 0 and B - move >= 0:
            return "First"

    # Check if the second player can force a win
    for move in B_moves:
        if B - move <= 0 and A <= 0:
            return "Second"
        if B - move >= 0 and A <= 0:
            return "Second"
        if B <= 0 and A - move >= 0:
            return "Second"

    # If no player can force a win, the second player wins
    return "Second"


# Read input
A, B = 2, 2
A_moves = [1, 2]
B_moves = [1]

# Find the winner
winner = find_winner(A, B, A_moves, B_moves)

# Print the winner
print(winner)
