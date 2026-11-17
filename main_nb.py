# To do:
# - implement batching
import marimo

__generated_with = "0.24.2"
app = marimo.App()

with app.setup:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    rng = np.Randon.RandomState(42)


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

    def emeans(X: np.ndarray, k: int, epochs: int, batch_size: int, revical_threshold: float):
        C0 = X[rng.choice(len(X), size=k, replace=False)]
        N = np.full(k, 1e-10, dtype=np.float32)
        S = C0 * N[:, None]
        t = 0
        a_0 = 1
        for i in range(epochs):
            n = len(X)
            M = epochs * np.floor(n/batch_size)
            for i in range(M):
                t += 1
                eps = 0.1
                a_step = a_0 * \
                    (eps + (0.5 * (1 - eps) * (1 + np.cos((np.pi * t)/M))))
                gamma_t = 1 - a_step

                nearest_centroids =


if __name__ == "__main__":
    app.run()
