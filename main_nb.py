# To do:
# - implement batching so X_b is the batch for each epoch
# - implement revive_dead_clusters

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
    return (X_coords, y_true)


@app.cell
def plot(X_coords, y_true):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(X_coords[:, 0], X_coords[:, 1], s=8, alpha=0.6)
    ax.set_xlabel("x axis")
    ax.set_ylabel("y axis")
    ax.axis("equal")
    plt.show()
    return (fig,)


@app.cell
def implementation(X_coords):
    X = X_coords
    k = 31

    def emeans(X: np.ndarray, k: int, T: int, B: int, tau: float):
        C0 = X[rng.choice(len(X), size=k, replace=False)]
        N = np.full(k, 1e-10, dtype=np.float32)
        S = C0 * N[:, None]
        t = 0
        a_0 = 1
        a_all = np.empty(len(X), dtype=np.int32)
        d = X.shape[1]
        n = np.zeros(k, np.float32)
        s = np.zeros((k, d), np.float32)
        for epoch in range(T):
            # n = len(X)
            M = T * np.floor(n/B)
            for i in range(M):
                t += 1
                eps = 0.1
                a_step = a_0 * \
                    (eps + (0.5 * (1 - eps) * (1 + np.cos((np.pi * t)/M))))
                gamma_t = 1 - a_step
                c = S / N[:, None]
                c_norms = (c ** 2).sum(axis=1)
                a = (c_norms[None, :] - 2 * X_b @ c.T).argmin(axis=1)
                n = np.bincount(a, minlength=k).astype(np.float32)
                # s = np.zeros((k, d), np.float32)
                s.fill(0)
                np.add.at(s, a, X_b)
                N = gamma_t * N + (1 - gamma_t) * n
                S = gamma_t * S + (1 - gamma_t) * s

                if t % 5 == 0:
                    S, N = revive_dead_clusters(tau, t, S, N)

        c_final = s / np.maximum(n, 1)[:, None]
        return c_final


if __name__ == "__main__":
    app.run()
