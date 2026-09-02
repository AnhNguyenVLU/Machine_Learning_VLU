"""
Sinh ảnh động GIF cho Lab 03 Naive Bayes.

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


def save_gif(ani, name, fps=10):
    path = os.path.join(OUT, name)
    ani.save(path, writer=PillowWriter(fps=fps))
    plt.close("all")
    print(f"wrote {name}  ({os.path.getsize(path)/1024:.0f} KB)")


def anim_posterior_cong_don():
    """Đọc văn bản từng từ một, xem log-odds cộng dồn và xác suất đổi theo."""
    # log P(tu | positive) - log P(tu | negative), số dương nghiêng về positive
    words = ["khoa", "hoc", "may", "rat", "hay", "nhung", "bai", "tap",
             "thi", "kho", "va", "chan", "qua", "khong", "thich"]
    contrib = [0.10, 0.05, 0.15, 0.55, 1.35, -0.35, 0.10, 0.05,
               -0.25, -0.95, 0.00, -1.25, -0.45, -0.85, 0.70]
    prior = 0.25                                   # log-odds tiên nghiệm
    run = prior + np.cumsum([0.0] + contrib)       # run[0] chỉ có prior
    prob = 1 / (1 + np.exp(-run))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 4.3),
                                 gridspec_kw={"width_ratios": [1.25, 1]})

    a1.axhline(0, color="k", lw=1.4)
    a1.set_xlim(-.7, len(words) - .3)
    a1.set_ylim(min(run.min(), -1.2) - .6, max(run.max(), 1.2) + .9)
    a1.set_xticks(range(len(words)))
    a1.set_xticklabels(words, rotation=45, ha="right", fontsize=8.5)
    a1.set_ylabel("log-odds cộng dồn")
    a1.set_title("Mỗi từ đẩy quyết định về một phía", fontsize=10.5)
    a1.text(.02, .96, "trên đường 0: nghiêng positive\ndưới đường 0: nghiêng negative",
            transform=a1.transAxes, va="top", fontsize=8.4, bbox=BOX)
    bars = a1.bar(range(len(words)), [0] * len(words), width=.62, color="0.85")
    curve, = a1.plot([], [], "o-", color=C4, lw=2.0, ms=4.5, zorder=5)

    a2.set_xlim(0, 1); a2.set_ylim(0, 1)
    a2.axis("off")
    a2.set_title("Xác suất là bài đánh giá tích cực", fontsize=10.5)
    gauge_bg = plt.Rectangle((.1, .42), .8, .14, facecolor="0.9", edgecolor="0.6")
    a2.add_patch(gauge_bg)
    gauge = plt.Rectangle((.1, .42), 0, .14, facecolor=C3)
    a2.add_patch(gauge)
    a2.plot([.5, .5], [.38, .60], color="k", lw=1.4)
    a2.text(.5, .34, "0.5", ha="center", fontsize=8.5)
    a2.text(.1, .60, "negative", fontsize=9, color=C2)
    a2.text(.9, .60, "positive", fontsize=9, color=C3, ha="right")
    ptxt = a2.text(.5, .72, "", ha="center", fontsize=13, fontweight="bold")
    wtxt = a2.text(.5, .22, "", ha="center", fontsize=10, bbox=BOX)

    HOLD = 10
    frames = list(range(len(run))) + [len(run) - 1] * HOLD

    def draw(i):
        for j, bar in enumerate(bars):
            if j < i:
                bar.set_height(contrib[j])
                bar.set_color(C3 if contrib[j] >= 0 else C2)
            else:
                bar.set_height(0)
        curve.set_data(range(i), run[:i] if i > 0 else [])
        p = prob[i]
        gauge.set_width(.8 * p)
        gauge.set_facecolor(C3 if p >= .5 else C2)
        ptxt.set_text(f"P(positive) = {p:.3f}")
        ptxt.set_color(C3 if p >= .5 else C2)
        if i == 0:
            wtxt.set_text("mới chỉ có prior, chưa đọc từ nào")
        else:
            w = words[i - 1]; c = contrib[i - 1]
            huong = "nghiêng positive" if c > 0 else ("nghiêng negative" if c < 0 else "trung tính")
            wtxt.set_text(f'vừa đọc "{w}"   ({c:+.2f}, {huong})')
        return bars, curve, gauge, ptxt, wtxt

    ani = FuncAnimation(fig, draw, frames=frames, blit=False)
    fig.suptitle("Naive Bayes cộng dồn bằng chứng theo từng từ", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, .94])
    save_gif(ani, "anim_posterior_cong_don.gif", fps=4)


if __name__ == "__main__":
    anim_posterior_cong_don()
    print("Xong.")
