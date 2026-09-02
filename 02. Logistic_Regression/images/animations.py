"""
Sinh ảnh động GIF cho Lab 02 Logistic Regression.

Chạy:  python animations.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({
    "figure.dpi": 84, "font.size": 10, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})
C1, C2, C3, C4 = "#2563eb", "#dc2626", "#059669", "#d97706"
BOX = dict(boxstyle="round,pad=0.32", facecolor="white", alpha=.93, edgecolor="0.8")


def save_gif(ani, name, fps=10):
    path = os.path.join(OUT, name)
    ani.save(path, writer=PillowWriter(fps=fps))
    plt.close("all")
    print(f"wrote {name}  ({os.path.getsize(path)/1024:.0f} KB)")


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def anim_logistic_hoc():
    """Ranh giới và trường xác suất của Logistic Regression hình thành dần qua từng bước GD."""
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=220, n_features=2, n_redundant=0,
                               n_clusters_per_class=1, class_sep=1.05,
                               flip_y=.02, random_state=4)
    X = (X - X.mean(0)) / X.std(0)

    lr, n_step = 0.62, 38
    w = np.array([-1.6, 1.9]); b = 1.4
    Ws, Bs, Ls = [w.copy()], [b], []
    for _ in range(n_step):
        p = sigmoid(X @ w + b)
        Ls.append(float(-np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))))
        w -= lr * (X.T @ (p - y)) / len(y)
        b -= lr * float((p - y).mean())
        Ws.append(w.copy()); Bs.append(b)

    pad = .8
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, 100),
                         np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, 100))
    G = np.c_[xx.ravel(), yy.ravel()]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.8, 4.3))
    a1.set_xlabel("$x_1$"); a1.set_ylabel("$x_2$")
    a1.set_title("Trường xác suất và ranh giới", fontsize=10.5)
    a2.set_xlabel("bước"); a2.set_ylabel("BCE")
    a2.set_title("Hàm mất mát", fontsize=10.5)
    a2.set_xlim(0, n_step); a2.set_ylim(0, max(Ls) * 1.12)
    lossline, = a2.plot([], [], color=C3, lw=2.2)
    info = a2.text(.97, .95, "", transform=a2.transAxes, ha="right", va="top",
                   fontsize=9.5, bbox=BOX)
    holder = {"cf": None, "cs": None}
    HOLD = 8
    frames = list(range(len(Ws))) + [len(Ws) - 1] * HOLD

    def draw(i):
        for key in ("cf", "cs"):
            if holder[key] is not None:
                holder[key].remove()
        P = sigmoid(G @ Ws[i] + Bs[i]).reshape(xx.shape)
        holder["cf"] = a1.contourf(xx, yy, P, levels=np.linspace(0, 1, 11),
                                   cmap="RdBu_r", alpha=.68, zorder=1)
        holder["cs"] = a1.contour(xx, yy, P, levels=[.5], colors="k",
                                  linewidths=2.4, zorder=2)
        a1.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu_r", s=20,
                   edgecolor="k", linewidth=.35, zorder=3)
        k = min(i, len(Ls) - 1)
        lossline.set_data(range(1, k + 2), Ls[:k + 1])
        acc = ((sigmoid(X @ Ws[i] + Bs[i]) >= .5).astype(int) == y).mean() * 100
        info.set_text(f"bước {i}\nBCE = {Ls[k]:.3f}\nacc = {acc:.1f}%")
        return lossline, info

    ani = FuncAnimation(fig, draw, frames=frames, blit=False)
    fig.suptitle("Logistic Regression học ranh giới qua từng bước", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, .94])
    save_gif(ani, "anim_logistic_hoc.gif", fps=9)


if __name__ == "__main__":
    anim_logistic_hoc()
    print("Xong.")
