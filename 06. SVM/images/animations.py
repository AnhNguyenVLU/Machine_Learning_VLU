"""
Sinh ảnh động (GIF) minh hoạ cho Lab 06 - Support Vector Machine.

Chạy:  python animations.py
Kết quả: các file .gif trong cùng thư mục, được nhúng vào notebook bằng markdown.
Tách riêng khỏi figures.py vì ảnh động chạy lâu hơn hình tĩnh khá nhiều.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({
    "figure.dpi": 90, "savefig.dpi": 90,
    "font.size": 10, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})
C1, C2, C3, C4 = "#2563eb", "#dc2626", "#059669", "#d97706"


def save_gif(ani, name, fps=8):
    p = os.path.join(OUT, name)
    ani.save(p, writer=PillowWriter(fps=fps))
    plt.close(ani._fig)
    mb = os.path.getsize(p) / 1e6
    print(f"wrote {name}  ({mb:.2f} MB)")


# ---------------------------------------------------------------- 1
def anim_gamma_rbf(n_frames=42, n_hold=8, grid=110):
    """Quét gamma của RBF theo thang log: ranh giới từ gần thẳng tới bao từng điểm."""
    X, y = make_moons(n_samples=260, noise=.24, random_state=42)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.35, random_state=0,
                                          stratify=y)
    gammas = np.logspace(np.log10(0.05), np.log10(200), n_frames)

    pad = .55
    gx = np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, grid)
    gy = np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, grid)
    GX, GY = np.meshgrid(gx, gy)
    P = np.c_[GX.ravel(), GY.ravel()]

    # tính trước toàn bộ khung để lúc render không phải fit lại model
    frames = []
    for g in gammas:
        clf = SVC(kernel="rbf", C=1.0, gamma=g).fit(Xtr, ytr)
        Z = clf.decision_function(P).reshape(GX.shape)
        frames.append((g, Z, len(clf.support_vectors_),
                       clf.score(Xtr, ytr), clf.score(Xte, yte)))
    frames += [frames[-1]] * n_hold          # giữ khung cuối lâu hơn để kịp đọc

    fig, ax = plt.subplots(figsize=(7.4, 5.4))

    def draw(i):
        g, Z, nsv, tr, te = frames[i]
        ax.clear()
        ax.grid(True, alpha=.25)
        ax.contourf(GX, GY, np.sign(Z), levels=[-2, 0, 2],
                    colors=[C1, C2], alpha=.13, zorder=0)
        ax.contour(GX, GY, Z, levels=[0], colors="k", linewidths=2.0, zorder=2)
        ax.scatter(Xtr[ytr == 0, 0], Xtr[ytr == 0, 1], s=20, c=C1, marker="o",
                   edgecolor="k", linewidth=.4, zorder=3, label="lớp $-1$ (train)")
        ax.scatter(Xtr[ytr == 1, 0], Xtr[ytr == 1, 1], s=20, c=C2, marker="s",
                   edgecolor="k", linewidth=.4, zorder=3, label="lớp $+1$ (train)")
        ax.scatter(Xte[:, 0], Xte[:, 1], s=16, facecolors="none",
                   edgecolors="#555555", linewidth=.6, zorder=4, label="tập test")
        ax.text(.015, .985,
                f"gamma = {g:7.3f}\n"
                f"support vector = {nsv}\n"
                f"train acc = {tr*100:5.1f}%\n"
                f"test  acc = {te*100:5.1f}%",
                transform=ax.transAxes, va="top", ha="left", fontsize=10,
                family="monospace", zorder=6,
                bbox=dict(boxstyle="round,pad=.4", fc="white", ec="lightgray",
                          alpha=.95))
        ax.set_xlim(gx[0], gx[-1]); ax.set_ylim(gy[0], gy[-1])
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
        ax.legend(fontsize=8, loc="lower right", framealpha=.95)
        ax.set_title("Ảnh hưởng của gamma trong RBF", fontweight="bold")

    ani = FuncAnimation(fig, draw, frames=len(frames), interval=130)
    ani._fig = fig
    save_gif(ani, "anim_gamma_rbf.gif", fps=8)


if __name__ == "__main__":
    anim_gamma_rbf()
    print("Xong.")
