"""
Sinh ảnh động GIF cho Lab 08 K-Means.

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
    "figure.dpi": 88, "font.size": 10, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})
C1, C2, C3, C4 = "#2563eb", "#dc2626", "#059669", "#d97706"
BOX = dict(boxstyle="round,pad=0.32", facecolor="white", alpha=.93, edgecolor="0.8")
PALETTE = ["#2563eb", "#dc2626", "#059669", "#d97706"]


def save_gif(ani, name, fps=10):
    path = os.path.join(OUT, name)
    ani.save(path, writer=PillowWriter(fps=fps))
    plt.close("all")
    print(f"wrote {name}  ({os.path.getsize(path)/1024:.0f} KB)")


def anim_lloyd():
    """Lloyd lặp Assign rồi Update. Mỗi nửa bước là một khung, inertia giảm dần."""
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=320, centers=4, cluster_std=1.15, random_state=7)
    rng = np.random.default_rng(3)
    K = 4
    C = X[rng.choice(len(X), K, replace=False)] + rng.normal(0, 2.6, (K, 2))

    states = []                                   # (tâm, nhãn hoặc None, tên bước, inertia)
    states.append((C.copy(), None, "Khởi tạo", None))
    for it in range(7):
        d = ((X[:, None, :] - C[None]) ** 2).sum(-1)
        lab = d.argmin(1)
        inertia = d[np.arange(len(X)), lab].sum()
        states.append((C.copy(), lab.copy(), f"Vòng {it + 1}: Assign", inertia))
        newC = np.array([X[lab == k].mean(0) if (lab == k).any() else C[k] for k in range(K)])
        moved = np.abs(newC - C).max()
        C = newC
        d2 = ((X[:, None, :] - C[None]) ** 2).sum(-1)
        inertia2 = d2[np.arange(len(X)), lab].sum()
        states.append((C.copy(), lab.copy(), f"Vòng {it + 1}: Update", inertia2))
        if moved < 1e-4:
            break

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4),
                                 gridspec_kw={"width_ratios": [1.35, 1]})
    a1.set_xlabel("$x_1$"); a1.set_ylabel("$x_2$")
    pad = 1.0
    a1.set_xlim(X[:, 0].min() - pad, X[:, 0].max() + pad)
    a1.set_ylim(X[:, 1].min() - pad, X[:, 1].max() + pad)
    inert = [s[3] for s in states if s[3] is not None]
    a2.set_xlim(0, len(inert) + 1); a2.set_ylim(0, max(inert) * 1.15)
    a2.set_xlabel("nửa bước"); a2.set_ylabel("inertia")
    a2.set_title("Inertia theo từng nửa bước", fontsize=10.5)
    iline, = a2.plot([], [], "o-", color=C4, lw=2.0, ms=4.5)
    itxt = a2.text(.97, .95, "", transform=a2.transAxes, ha="right", va="top",
                   fontsize=9.5, bbox=BOX)
    holder = {"pts": None, "cen": None}
    HOLD = 10
    frames = list(range(len(states))) + [len(states) - 1] * HOLD

    def draw(i):
        Cn, lab, name, _ = states[i]
        for k in ("pts", "cen"):
            if holder[k] is not None:
                holder[k].remove()
        cols = ["0.72"] * len(X) if lab is None else [PALETTE[l] for l in lab]
        holder["pts"] = a1.scatter(X[:, 0], X[:, 1], c=cols, s=16, alpha=.8, zorder=2)
        holder["cen"] = a1.scatter(Cn[:, 0], Cn[:, 1], marker="*", s=420,
                                   c=PALETTE[:len(Cn)], edgecolor="k",
                                   linewidth=1.2, zorder=5)
        a1.set_title(name, fontsize=10.5)
        got = [s[3] for s in states[:i + 1] if s[3] is not None]
        iline.set_data(range(1, len(got) + 1), got)
        itxt.set_text(f"inertia = {got[-1]:.0f}" if got else "chưa gán nhãn")
        return holder["pts"], holder["cen"], iline, itxt

    ani = FuncAnimation(fig, draw, frames=frames, blit=False)
    fig.suptitle("Thuật toán Lloyd của K-Means", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, .94])
    save_gif(ani, "anim_lloyd.gif", fps=2)


if __name__ == "__main__":
    anim_lloyd()
    print("Xong.")
