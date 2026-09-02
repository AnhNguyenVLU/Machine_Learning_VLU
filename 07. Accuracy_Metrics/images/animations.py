"""
Sinh ảnh động (GIF) minh hoạ cho Lab 07 - Đánh giá mô hình phân loại.

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
from sklearn.metrics import roc_curve, roc_auc_score

OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({
    "figure.dpi": 90, "savefig.dpi": 90,
    "font.size": 10, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})
C1, C2, C3, C4 = "#2563eb", "#dc2626", "#059669", "#d97706"

# dùng lại đúng bộ điểm số của figures.py để GIF khớp với hình 02 và 03
RNG = np.random.default_rng(7)
NEG = RNG.normal(0.32, 0.135, 4000)
POS = RNG.normal(0.63, 0.135, 4000)


def save_gif(ani, name, fps=8):
    p = os.path.join(OUT, name)
    ani.save(p, writer=PillowWriter(fps=fps))
    plt.close(ani._fig)
    mb = os.path.getsize(p) / 1e6
    print(f"wrote {name}  ({mb:.2f} MB)")


def _pdf(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))


# ---------------------------------------------------------------- 1
def anim_threshold_roc(n_frames=44, n_hold=8):
    """Ngưỡng trượt bên trái, chấm tương ứng chạy trên đường ROC bên phải."""
    y = np.r_[np.zeros(len(NEG)), np.ones(len(POS))]
    s = np.r_[NEG, POS]
    fpr, tpr, _ = roc_curve(y, s)
    auc = roc_auc_score(y, s)

    ts = np.linspace(0.95, 0.05, n_frames)   # ngưỡng đi từ cao xuống thấp
    stats = []
    for t in ts:
        TP = float((POS >= t).sum()); FN = float((POS < t).sum())
        FP = float((NEG >= t).sum()); TN = float((NEG < t).sum())
        stats.append((t, FP / (FP + TN), TP / (TP + FN),
                      TP / max(TP + FP, 1.0), TP / (TP + FN)))
    stats += [stats[-1]] * n_hold            # giữ khung cuối lâu hơn để kịp đọc

    x = np.linspace(0, 1, 400)
    pn, pp = _pdf(x, .32, .135), _pdf(x, .63, .135)

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8))
    fig.subplots_adjust(left=.07, right=.98, top=.86, bottom=.14, wspace=.24)

    def draw(i):
        t, f, r, prec, rec = stats[i]

        ax = axes[0]
        ax.clear()
        ax.grid(True, alpha=.25)
        ax.fill_between(x, 0, pn, where=x < t, color=C1, alpha=.45)
        ax.fill_between(x, 0, pn, where=x >= t, color=C4, alpha=.75)
        ax.fill_between(x, 0, pp, where=x < t, color="#7c3aed", alpha=.75)
        ax.fill_between(x, 0, pp, where=x >= t, color=C2, alpha=.45)
        ax.plot(x, pn, color=C1, lw=1.6)
        ax.plot(x, pp, color=C2, lw=1.6)
        ax.axvline(t, color="k", lw=2.6)
        ax.text(t, 4.08, f" ngưỡng = {t:.2f}", fontsize=10, fontweight="bold",
                ha="left" if t < .6 else "right", va="center")
        ax.set_xlim(0, 1); ax.set_ylim(0, 4.4)
        ax.set_xlabel("điểm số model chấm cho mẫu")
        ax.set_ylabel("mật độ mẫu")
        ax.set_title("Ngưỡng trượt trên hai phân phối", fontsize=10.5)

        ax = axes[1]
        ax.clear()
        ax.grid(True, alpha=.25)
        ax.plot(fpr, tpr, color=C1, lw=2.4, label=f"ROC (AUC = {auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="đoán mò (AUC = 0.5)")
        ax.fill_between(fpr, 0, tpr, color=C1, alpha=.10)
        ax.plot([f, f], [0, r], color=C3, lw=1, ls=":")
        ax.plot([0, f], [r, r], color=C3, lw=1, ls=":")
        ax.scatter([f], [r], s=150, color=C2, edgecolor="k", linewidth=.9, zorder=6)
        ax.text(.985, .035,
                f"ngưỡng   = {t:.2f}\n"
                f"TPR      = {r:.3f}\n"
                f"FPR      = {f:.3f}\n"
                f"precision= {prec:.3f}\n"
                f"recall   = {rec:.3f}",
                transform=ax.transAxes, va="bottom", ha="right", fontsize=9.5,
                family="monospace",
                bbox=dict(boxstyle="round,pad=.4", fc="white", ec="lightgray",
                          alpha=.95))
        ax.set_xlim(-.02, 1.02); ax.set_ylim(-.02, 1.05)
        ax.set_xlabel("FPR = FP/(FP+TN)")
        ax.set_ylabel("TPR = TP/(TP+FN) = Recall")
        ax.legend(fontsize=8.5, loc="upper left", framealpha=.95)
        ax.set_title("Chấm tương ứng chạy trên đường ROC", fontsize=10.5)

        fig.suptitle("Từ ngưỡng tới đường ROC", fontweight="bold", fontsize=12)

    ani = FuncAnimation(fig, draw, frames=len(stats), interval=130)
    ani._fig = fig
    save_gif(ani, "anim_nguong_va_roc.gif", fps=8)


if __name__ == "__main__":
    anim_threshold_roc()
    print("Xong.")
