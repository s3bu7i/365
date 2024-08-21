// #include <iostream>
// #include <vector>
// using namespace std;

// string find_winner(int A, int B, const vector<int>& A_moves, const vector<int>& B_moves) {
//     // Initialize DP table
//     vector<vector<bool>> dp(A + 1, vector<bool>(B + 1, false));

//     // Fill the DP table
//     for (int i = 0; i <= A; ++i) {
//         for (int j = 0; j <= B; ++j) {
//             // Check if the current player can force a win
//             for (int move : A_moves) {
//                 if (i - move >= 0 && !dp[i - move][j]) {
//                     dp[i][j] = true;
//                 }
//             }
//             for (int move : B_moves) {
//                 if (j - move >= 0 && !dp[i][j - move]) {
//                     dp[i][j] = true;
//                 }
//             }
//         }
//     }

//     // Determine the winner from the initial state
//     return dp[A][B] ? "First" : "Second";
// }

// int main() {
//     // Read input
//     int A, B;
//     cin >> A >> B;

//     int K;
//     cin >> K;
//     vector<int> A_moves(K);
//     for (int i = 0; i < K; ++i) {
//         cin >> A_moves[i];
//     }

//     int L;
//     cin >> L;
//     vector<int> B_moves(L);
//     for (int i = 0; i < L; ++i) {
//         cin >> B_moves[i];
//     }

//     // Find the winner
//     string winner = find_winner(A, B, A_moves, B_moves);

//     // Print the winner
//     cout << winner << endl;

//     return 0;
// }

#include <iostream>
#include <vector>
#include <stack>
#include <cstring>

using namespace std;

const int MAXN = 300000;

vector<int> adj[MAXN];
int result[MAXN];
bool visited[MAXN];
bool inStack[MAXN];

void dfs(int v) {
    stack<int> s;
    s.push(v);
    while (!s.empty()) {
        int u = s.top();
        if (!visited[u]) {
            visited[u] = true;
            inStack[u] = true;
            for (int neighbor : adj[u]) {
                if (!visited[neighbor]) {
                    s.push(neighbor);
                }
            }
        } else {
            s.pop();
            inStack[u] = false;
            bool canMove = false;
            for (int neighbor : adj[u]) {
                if (result[neighbor] == 2) {
                    canMove = true;
                    break;
                }
            }
            if (canMove) {
                result[u] = 1; // FIRST wins
            } else {
                result[u] = 2; // SECOND wins
            }
        }
    }
}

void solve_game(int n, int m, vector<pair<int, int>>& edges) {
    for (int i = 0; i < n; ++i) {
        adj[i].clear();
        result[i] = 0;
        visited[i] = false;
        inStack[i] = false;
    }

    for (auto& edge : edges) {
        adj[edge.first - 1].push_back(edge.second - 1);
    }

    for (int i = 0; i < n; ++i) {
        if (!visited[i]) {
            dfs(i);
        }
    }

    for (int i = 0; i < n; ++i) {
        if (result[i] == 1) {
            cout << "FIRST" << endl;
        } else if (result[i] == 2) {
            cout << "SECOND" << endl;
        } else {
            cout << "DRAW" << endl;
        }
    }
    cout << endl;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    while (cin >> n >> m) {
        vector<pair<int, int>> edges(m);
        for (int i = 0; i < m; ++i) {
            cin >> edges[i].first >> edges[i].second;
        }
        solve_game(n, m, edges);
    }

    return 0;
}