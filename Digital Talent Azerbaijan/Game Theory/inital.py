def winner(n):

  if n <= 0:
    return 2
  if n <= 3:
    return 1 

  dp = [0] * (n + 1)
  dp[1] = dp[2] = dp[3] = 1

  for i in range(4, n + 1):
    if dp[i - 1] == 0 or dp[i - 2] == 0 or (i % 3 == 2 and dp[i - 3] == 0):
      dp[i] = 1 
    else:
      dp[i] = 0 
  return dp[n]


n = 1
print(winner(n))
