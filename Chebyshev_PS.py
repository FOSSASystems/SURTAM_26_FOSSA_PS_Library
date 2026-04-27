import numpy as np



# ============================================================
# 1. CHEBYSHEV NODES
# ============================================================

def chebyshev_lobatto_nodes(N):
    i = np.arange(N + 1)
    tau = np.cos(np.pi * (N - i) / N)
    return tau


# ============================================================
# 2. DIFFERENTIATION MATRIX
# ============================================================

def chebyshev_diff_matrix(N, tau):
    D = np.zeros((N + 1, N + 1))
    c = np.ones(N + 1)
    c[0] = 2
    c[-1] = 2

    for k in range(N + 1):
        for j in range(N + 1):
            if k != j:
                D[k, j] = -(c[k] / c[j]) * ((-1) ** (k + j)) / (tau[j] - tau[k])
            else:
                if k == 0:
                    D[k, j] = -(2 * N**2 + 1) / 6
                elif k == N:
                    D[k, j] = (2 * N**2 + 1) / 6
                else:
                    D[k, j] = -tau[k] / (2 * (1 - tau[k] ** 2))
    return D


# ============================================================
# 3. CLENSHAW-CURTIS WEIGHTS (CORRECTOS)
# ============================================================

def clenshaw_curtis_weights(N):
    w = np.zeros(N + 1)

    if N % 2 == 0:
        w[0] = 1.0 / (N**2 - 1)
        w[N] = w[0]

        for s in range(1, N//2 + 1):
            total = 0.0
            for j in range(0, N//2 + 1):
                a_j = 0.5 if (j == 0 or j == N//2) else 1.0
                total += (a_j / (1 - 4*j**2)) * np.cos(2 * np.pi * j * s / N)

            w[s] = (4.0 / N) * total
            w[N - s] = w[s]

    else:
        w[0] = 1.0 / (N**2)
        w[N] = w[0]

        for s in range(1, (N - 1)//2 + 1):
            total = 0.0
            for j in range(0, (N - 1)//2 + 1):
                a_j = 0.5 if (j == 0 or j == (N - 1)//2) else 1.0
                total += (a_j / (1 - 4*j**2)) * np.cos(2 * np.pi * j * s / N)

            w[s] = (4.0 / N) * total
            w[N - s] = w[s]

    return w