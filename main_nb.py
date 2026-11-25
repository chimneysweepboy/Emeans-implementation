# To do:
# - remove the hardcoded 3100


import marimo

__generated_with = "0.24.2"
app = marimo.App()

with app.setup:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(42)


@app.cell
def import_data():
    df = pd.read_csv("data.txt", sep="\t", header=None,
                     names=["X", "Y", "Cluster"])
    # print(df.head())
    X_coords = df[["X", "Y"]].to_numpy()
    y_true = df["Cluster"].to_numpy(dtype=np.float32)
    # print(X_coords)
    return (X_coords,)


@app.cell
def plot(X_coords):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(X_coords[:, 0], X_coords[:, 1], s=8, alpha=0.6)
    ax.set_xlabel("x axis")
    ax.set_ylabel("y axis")
    ax.axis("equal")
    plt.show()
    return


@app.cell
def implementation(X_coords):
    X = X_coords
    k = 64
    no_points = 3100
    T = 5
    B = 0.035 * (k ** 0.13) * (no_points ** 0.88) * (T ** 0.81)
    B = 2 ** round(np.log2(B))

    def revive_dead_clusters(tau: float, t: int, S: np.ndarray, N: np.ndarray):
        delta = 2.0 ** -7
        mean = N.mean()
        dead = np.flatnonzero(N < tau * mean)
        live = np.flatnonzero(N >= tau * mean)
        if dead.size == 0 or live.size == 0:
            return S, N
        no_revive = min(dead.size, live.size)
        dead = dead[:no_revive]
        p = N[live].astype(np.float64)
        p /= p.sum()
        donors = rng.choice(live, size=no_revive, replace=False, p=p)
        c = S[donors] / N[donors, None]
        signs = np.ones(c.shape)
        signs[:, 1::2] = -1.0
        half = 0.5 * N[donors]
        N[dead] = half
        S[dead] = c * (1 + delta * signs) * half[:, None]
        N[donors] = half
        S[donors] = c * (1 - delta * signs) * half[:, None]
        return S, N

    def emeans(X: np.ndarray, k: int, T: int, B: int, tau: float):
        C0 = X[rng.choice(len(X), size=k, replace=False)]
        N = np.full(k, 1e-10, dtype=np.float32)
        S = C0 * N[:, None]
        t = 0
        a_0 = np.minimum(0.824 * (k ** -0.46) * (B ** 0.9)
                         * (3100 ** -0.45) * (T ** -0.11), 1)
        d = X.shape[1]
        n_external = np.zeros(k, np.float32)
        s_external = np.zeros((k, d), np.float32)
        M = T * (no_points // B)
        for epoch in range(T):
            # n = len(X)
            idx = rng.permutation(no_points)
            for i in range(0, no_points - B + 1, B):
                X_b = X[idx[i:i + B]]
                eps = 0.1
                a_step = a_0 * \
                    (eps + (0.5 * (1 - eps) * (1 + np.cos((np.pi * t)/M))))
                gamma_t = 1 - a_step
                c = S / N[:, None]
                c_norms = (c ** 2).sum(axis=1)
                a = (c_norms[None, :] - 2 * X_b @ c.T).argmin(axis=1)
                n = np.bincount(a, minlength=k).astype(np.float32)
                s = np.zeros((k, d), np.float32)
                np.add.at(s, a, X_b)

                N = gamma_t * N + (1 - gamma_t) * n
                S = gamma_t * S + (1 - gamma_t) * s

                if epoch == T - 1:
                    n_external += n
                    s_external += s

                t += 1

                if t % 5 == 0:
                    S, N = revive_dead_clusters(tau, t, S, N)

        c_final = s_external / np.maximum(n_external, 1)[:, None]
        return c_final

    c = emeans(X, k, T, B, tau=0.1)  # tau is arbitrary for now
    d2 = ((X[:, None, :] - c[None, :, :]) ** 2).sum(-1)
    print("MSE: ", d2.min(axis=1).mean())
    return (c,)


@app.cell
def assignments(X_coords, c):
    dist = ((X_coords[:, None, :] - c[None, :, :]) ** 2).sum(-1)
    assign = dist.argmin(axis=1)

    _fig, _ax = plt.subplots(figsize=(7, 6))
    _ax.scatter(X_coords[:, 0], X_coords[:, 1], c=assign, cmap="tab20",
                s=8, alpha=0.8)
    _ax.scatter(c[:, 0], c[:, 1], marker="X", s=70, c="black")
    _ax.set_aspect("equal")
    _fig.savefig("clusters.png", dpi=150, bbox_inches="tight")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
