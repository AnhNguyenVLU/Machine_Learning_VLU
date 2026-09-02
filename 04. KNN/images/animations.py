"""
Sinh ảnh động (GIF) cho Lab 04 - K-Nearest Neighbors (KNN).

Chạy:  python animations.py
Kết quả: các file .gif trong cùng thư mục, được nhúng vào notebook bằng markdown.
Sinh viên có thể sửa script này để tự thí nghiệm (đổi dải k, đổi mức nhiễu của
dữ liệu, đổi metric...) rồi chạy lại để xem ảnh động thay đổi thế nào.

Lưu ý: GIF được cố ý giữ nhỏ (dpi thấp, lưới contour thưa) để notebook không
phình ra. Muốn hình nét hơn thì tăng "figure.dpi" và giảm bước lưới h.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import ListedColormap

OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({
    "figure.dpi": 90, "savefig.dpi": 90,
    "font.size": 10, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})
C1, C2, C3, C4 = "#2563eb", "#dc2626", "#059669", "#d97706"
CMAP_BG = ListedColormap(["#dbeafe", "#fee2e2"])


def save_gif(ani, name, fps=8):
    p = os.path.join(OUT, name)
    ani.save(p, writer=animation.PillowWriter(fps=fps))
    plt.close("all")
    print("wrote", name, f"({os.path.getsize(p) / 1e6:.2f} MB)")


def _thanh_tien_do(fig):
    """Thanh tiến độ mảnh ở đáy hình, cho biết ảnh động đang chạy tới đâu."""
    from matplotlib.patches import Rectangle
    axp = fig.add_axes([0.04, 0.018, 0.92, 0.014])
    axp.set_xlim(0, 1); axp.set_ylim(0, 1)
    axp.set_xticks([]); axp.set_yticks([]); axp.grid(False)
    axp.set_facecolor("#e2e8f0")
    for sp in axp.spines.values():
        sp.set_visible(False)
    bar = Rectangle((0, 0), 0, 1, color=C4)
    axp.add_patch(bar)
    return bar


def anim_k_thay_doi():
    """Cho k chạy từ 1 lên 40: ranh giới đi từ lởm chởm sang trơn."""
    from sklearn.datasets import make_moons
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split

    X, y = make_moons(n_samples=400, noise=0.30, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.4, random_state=0,
                                          stratify=y)
    x0, x1 = X[:, 0].min() - .5, X[:, 0].max() + .5
    y0, y1 = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.045
    xx, yy = np.meshgrid(np.arange(x0, x1, h), np.arange(y0, y1, h))
    G = np.c_[xx.ravel(), yy.ravel()]

    ks = list(range(1, 41))
    Z, acc = [], []
    for k in ks:
        m = KNeighborsClassifier(n_neighbors=k).fit(Xtr, ytr)
        Z.append(m.predict(G).reshape(xx.shape))
        acc.append(m.score(Xte, yte))

    # giữ khung đầu và khung cuối lâu hơn để người xem kịp đọc
    idx = [0] * 3 + list(range(len(ks))) + [len(ks) - 1] * 8

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    fig.subplots_adjust(left=.04, right=.96, top=.86, bottom=.07)
    bar = _thanh_tien_do(fig)

    def draw(fi):
        bar.set_width((fi + 1) / len(idx))
        j = idx[fi]
        ax.clear()
        ax.pcolormesh(xx, yy, Z[j], cmap=CMAP_BG, shading="auto", zorder=0)
        ax.contour(xx, yy, Z[j], levels=[.5], colors="k", linewidths=1.6, zorder=1)
        ax.scatter(Xtr[ytr == 0, 0], Xtr[ytr == 0, 1], s=14, color=C1,
                   edgecolor="w", linewidth=.3, zorder=3)
        ax.scatter(Xtr[ytr == 1, 0], Xtr[ytr == 1, 1], s=14, color=C2, marker="s",
                   edgecolor="w", linewidth=.3, zorder=3)
        ax.set_xlim(x0, x1 - h); ax.set_ylim(y0, y1 - h)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_title(f"k = {ks[j]:2d}     test accuracy = {acc[j] * 100:.1f}%",
                     fontsize=11.5, color=C4 if ks[j] <= 3 else "#334155")
        return ()

    fig.suptitle("Ranh giới KNN khi k tăng dần", fontweight="bold", fontsize=12)
    ani = animation.FuncAnimation(fig, draw, frames=len(idx), interval=125)
    save_gif(ani, "anim_k_thay_doi.gif", fps=8)


if __name__ == "__main__":
    anim_k_thay_doi()
    print("Xong.")
