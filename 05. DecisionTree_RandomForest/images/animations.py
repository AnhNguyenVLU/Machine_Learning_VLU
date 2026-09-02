"""
Sinh ảnh động (GIF) cho Lab 05 - Decision Tree & Random Forest.

Chạy:  python animations.py
Kết quả: các file .gif trong cùng thư mục, được nhúng vào notebook bằng markdown.
Sinh viên có thể sửa script này để tự thí nghiệm (đổi dải max_depth, đổi mức
nhiễu của dữ liệu, đổi criterion...) rồi chạy lại để xem ảnh động thay đổi.

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


def save_gif(ani, name, fps=6):
    p = os.path.join(OUT, name)
    ani.save(p, writer=animation.PillowWriter(fps=fps))
    plt.close("all")
    print("wrote", name, f"({os.path.getsize(p) / 1e6:.2f} MB)")


def _thanh_tien_do(fig):
    """Thanh tiến độ mảnh ở đáy hình, cho biết ảnh động đang chạy tới đâu."""
    from matplotlib.patches import Rectangle
    axp = fig.add_axes([0.03, 0.022, 0.93, 0.016])
    axp.set_xlim(0, 1); axp.set_ylim(0, 1)
    axp.set_xticks([]); axp.set_yticks([]); axp.grid(False)
    axp.set_facecolor("#e2e8f0")
    for sp in axp.spines.values():
        sp.set_visible(False)
    bar = Rectangle((0, 0), 0, 1, color=C4)
    axp.add_patch(bar)
    return bar


def anim_cay_sau_dan():
    """Cho max_depth chạy từ 1 lên 10: ranh giới bậc thang mỗi lúc một chi tiết."""
    from sklearn.datasets import make_moons
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split

    X, y = make_moons(n_samples=500, noise=0.32, random_state=1)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.4, random_state=0,
                                          stratify=y)
    x0, x1 = X[:, 0].min() - .5, X[:, 0].max() + .5
    y0, y1 = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.045
    xx, yy = np.meshgrid(np.arange(x0, x1, h), np.arange(y0, y1, h))
    G = np.c_[xx.ravel(), yy.ravel()]

    depths = list(range(1, 11))
    Z, leaves, tr, te = [], [], [], []
    for d in depths:
        m = DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xtr, ytr)
        Z.append(m.predict(G).reshape(xx.shape))
        leaves.append(m.get_n_leaves())
        tr.append(m.score(Xtr, ytr) * 100)
        te.append(m.score(Xte, yte) * 100)
    best = int(np.argmax(te))

    # mỗi độ sâu giữ 3 khung, riêng khung cuối giữ thêm 9 khung để kịp đọc
    idx = [j for j in range(len(depths)) for _ in range(3)] + [len(depths) - 1] * 9

    fig, (axb, axc) = plt.subplots(1, 2, figsize=(9.4, 4.3),
                                   gridspec_kw=dict(width_ratios=[1.15, 1]))
    fig.subplots_adjust(left=.03, right=.96, top=.80, bottom=.18, wspace=.22)
    bar = _thanh_tien_do(fig)
    ymin, ymax = min(te) - 4, max(tr) + 4

    def draw(fi):
        j = idx[fi]
        bar.set_width((fi + 1) / len(idx))
        axb.clear()
        axb.pcolormesh(xx, yy, Z[j], cmap=CMAP_BG, shading="auto", zorder=0)
        axb.contour(xx, yy, Z[j], levels=[.5], colors="k", linewidths=1.6, zorder=1)
        axb.scatter(Xtr[ytr == 0, 0], Xtr[ytr == 0, 1], s=13, color=C1,
                    edgecolor="w", linewidth=.3, zorder=3)
        axb.scatter(Xtr[ytr == 1, 0], Xtr[ytr == 1, 1], s=13, color=C2, marker="s",
                    edgecolor="w", linewidth=.3, zorder=3)
        axb.set_xlim(x0, x1 - h); axb.set_ylim(y0, y1 - h)
        axb.set_xticks([]); axb.set_yticks([]); axb.grid(False)
        axb.set_title(f"max_depth = {depths[j]}   ({leaves[j]} lá)", fontsize=11)

        axc.clear()
        axc.grid(True, alpha=.25)
        axc.plot(depths[:j + 1], tr[:j + 1], "o-", color=C1, ms=4,
                 label="Train accuracy")
        axc.plot(depths[:j + 1], te[:j + 1], "s-", color=C2, ms=4,
                 label="Test accuracy")
        if j >= best:
            axc.axvline(depths[best], color=C3, ls="--", lw=1.6)
            axc.text(depths[best] + .18, ymin + 1.4,
                     f"test tốt nhất ở depth {depths[best]}", color=C3,
                     fontsize=8.4, va="bottom")
        axc.set_xlim(.5, 10.5); axc.set_ylim(ymin, ymax)
        axc.set_xticks(depths)
        axc.set_xlabel("max_depth"); axc.set_ylabel("Accuracy (%)")
        axc.legend(fontsize=8.4, loc="lower right", framealpha=.95)
        axc.set_title(f"train {tr[j]:.1f}%   test {te[j]:.1f}%", fontsize=11)
        return ()

    fig.suptitle("Cây sâu dần: ranh giới và accuracy", fontweight="bold",
                 fontsize=12)
    ani = animation.FuncAnimation(fig, draw, frames=len(idx), interval=167)
    save_gif(ani, "anim_cay_sau_dan.gif", fps=6)


if __name__ == "__main__":
    anim_cay_sau_dan()
    print("Xong.")
