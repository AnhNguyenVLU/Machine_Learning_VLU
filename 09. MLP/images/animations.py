"""
Sinh ảnh động GIF cho Lab 09 MLP.

Chạy:  python animations.py
Mạng được cài bằng numpy thuần nên script chạy được cả khi máy chưa có PyTorch.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({
    "figure.dpi": 86, "font.size": 10, "axes.grid": True, "grid.alpha": 0.25,
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


def anim_mlp_hoc():
    """MLP 2-16-16-1 học dữ liệu hai vành trăng. Ranh giới cong dần theo epoch."""
    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split

    X, y = make_moons(n_samples=400, noise=.22, random_state=42)
    X = (X - X.mean(0)) / X.std(0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.35, random_state=0, stratify=y)

    rng = np.random.default_rng(1)
    sizes = [2, 16, 16, 1]
    Ws = [rng.normal(0, np.sqrt(2 / sizes[i]), (sizes[i], sizes[i + 1]))
          for i in range(3)]
    bs = [np.zeros(sizes[i + 1]) for i in range(3)]

    def forward(A):
        cache = [A]
        for i in range(3):
            Z = A @ Ws[i] + bs[i]
            A = np.maximum(0, Z) if i < 2 else 1 / (1 + np.exp(-np.clip(Z, -500, 500)))
            cache.append(A)
        return cache

    def step(lr=.12):
        c = forward(Xtr)
        p = c[-1].ravel()
        d = ((p - ytr) / len(ytr))[:, None]
        for i in (2, 1, 0):
            gW = c[i].T @ d
            gb = d.sum(0)
            if i > 0:
                d = (d @ Ws[i].T) * (c[i] > 0)
            Ws[i] -= lr * gW
            bs[i] -= lr * gb
        return float(-np.mean(ytr * np.log(p + 1e-12) + (1 - ytr) * np.log(1 - p + 1e-12)))

    pad = .55
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, 110),
                         np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, 110))
    G = np.c_[xx.ravel(), yy.ravel()]

    checkpoints = [0, 5, 10, 20, 35, 55, 80, 120, 170, 240, 330, 450,
                   600, 800, 1050, 1350, 1700, 2100]
    snaps = []
    losses = []
    ep = 0
    for target in checkpoints:
        while ep < target:
            losses.append(step()); ep += 1
        P = forward(G)[-1].reshape(xx.shape)
        acc = ((forward(Xte)[-1].ravel() >= .5).astype(int) == yte).mean() * 100
        snaps.append((P.copy(), ep, losses[-1] if losses else np.nan, acc))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3),
                                 gridspec_kw={"width_ratios": [1.25, 1]})
    a1.set_xlabel("$x_1$"); a1.set_ylabel("$x_2$")
    a1.set_title("Ranh giới quyết định", fontsize=10.5)
    a2.set_xlabel("epoch"); a2.set_ylabel("BCE trên tập train")
    a2.set_title("Hàm mất mát", fontsize=10.5)
    a2.set_xscale("symlog"); a2.set_xlim(0, checkpoints[-1])
    a2.set_ylim(0, max(losses[:5]) * 1.1)
    lline, = a2.plot([], [], color=C3, lw=2.0)
    itxt = a2.text(.97, .95, "", transform=a2.transAxes, ha="right", va="top",
                   fontsize=9.5, bbox=BOX)
    holder = {"cf": None, "cs": None}
    HOLD = 8
    frames = list(range(len(snaps))) + [len(snaps) - 1] * HOLD

    def draw(i):
        P, ep_i, ls, acc = snaps[i]
        for k in ("cf", "cs"):
            if holder[k] is not None:
                holder[k].remove()
        holder["cf"] = a1.contourf(xx, yy, P, levels=np.linspace(0, 1, 11),
                                   cmap="RdBu_r", alpha=.68, zorder=1)
        holder["cs"] = a1.contour(xx, yy, P, levels=[.5], colors="k",
                                  linewidths=2.4, zorder=2)
        a1.scatter(Xtr[:, 0], Xtr[:, 1], c=ytr, cmap="RdBu_r", s=16,
                   edgecolor="k", linewidth=.3, zorder=3)
        lline.set_data(range(1, ep_i + 1), losses[:ep_i])
        itxt.set_text(f"epoch {ep_i}\nBCE = {ls:.3f}\ntest acc = {acc:.1f}%")
        return lline, itxt

    ani = FuncAnimation(fig, draw, frames=frames, blit=False)
    fig.suptitle("MLP học ranh giới cong trên dữ liệu hai vành trăng", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, .94])
    save_gif(ani, "anim_mlp_hoc.gif", fps=3)


if __name__ == "__main__":
    anim_mlp_hoc()
    print("Xong.")
