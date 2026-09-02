"""
Sinh ảnh động GIF cho Lab 01 Linear Regression.

Chạy:  python animations.py
Kết quả: file .gif trong cùng thư mục, được nhúng vào notebook bằng markdown.
Tách riêng khỏi figures.py vì chạy lâu hơn nhiều.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({
    "figure.dpi": 90, "font.size": 10, "axes.grid": True, "grid.alpha": 0.25,
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


def anim_gradient_descent():
    """Gradient Descent học dần đường hồi quy, xem song song với vị trí trên mặt mất mát."""
    rng = np.random.default_rng(42)
    N = 60
    xr = np.linspace(0, 10, N)
    x = (xr - xr.mean()) / xr.std()
    y = 3 * xr + 5 + rng.normal(0, 1.5, N)
    w_opt = (x * y).mean() / (x * x).mean()
    b_opt = y.mean()

    lr, n_step = 0.052, 48
    w, b = -4.0, 42.0
    P, L = [(w, b)], []
    for _ in range(n_step):
        e = w * x + b - y
        L.append((e ** 2).mean())
        w -= lr * 2 * (e * x).mean()
        b -= lr * 2 * e.mean()
        P.append((w, b))
    P = np.array(P); L = np.array(L)

    W, B = np.meshgrid(np.linspace(-16, 30, 200), np.linspace(-8, 48, 200))
    Z = np.mean((W[None] * x[:, None, None] + B[None] - y[:, None, None]) ** 2, axis=0)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.3))

    a1.scatter(xr, y, s=24, color=C1, alpha=.65, zorder=3)
    line, = a1.plot([], [], color=C2, lw=2.6, zorder=4)
    a1.set_xlabel("x"); a1.set_ylabel("y")
    a1.set_title("Đường hồi quy đang được học", fontsize=10.5)
    a1.set_ylim(y.min() - 6, y.max() + 6)
    a1.set_xlim(xr.min() - .4, xr.max() + .4)
    # đặt ở góc dưới phải: vùng này không bao giờ bị đường hồi quy hay dữ liệu đi qua
    info = a1.text(.97, .04, "", transform=a1.transAxes, ha="right", va="bottom",
                   fontsize=9.5, bbox=BOX)

    a2.contour(W, B, Z, levels=np.geomspace(Z.min() + 1, Z.max(), 14),
               colors="0.6", linewidths=.8)
    a2.scatter([w_opt], [b_opt], marker="*", s=280, color=C4,
               edgecolor="k", linewidth=.6, zorder=5)
    a2.annotate("nghiệm tối ưu", xy=(w_opt, b_opt), xytext=(.62, .16),
                textcoords="axes fraction", fontsize=8.6, color=C4, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C4, lw=1.3), bbox=BOX)
    trail, = a2.plot([], [], "-", color=C2, lw=1.6, zorder=4)
    head, = a2.plot([], [], "o", color=C2, ms=8, zorder=5)
    a2.set_xlim(-16, 30); a2.set_ylim(-8, 48)
    a2.set_xlabel("w"); a2.set_ylabel("b")
    a2.set_title("Vị trí trên mặt mất mát", fontsize=10.5)

    xs = np.linspace(xr.min(), xr.max(), 10)
    xs_s = (xs - xr.mean()) / xr.std()
    HOLD = 8                                  # giữ khung cuối cho người xem kịp đọc
    frames = list(range(len(P))) + [len(P) - 1] * HOLD

    def draw(i):
        wi, bi = P[i]
        line.set_data(xs, wi * xs_s + bi)
        trail.set_data(P[:i + 1, 0], P[:i + 1, 1])
        head.set_data([P[i, 0]], [P[i, 1]])
        info.set_text(f"bước {i}\nMSE = {L[min(i, len(L) - 1)]:.2f}")
        return line, trail, head, info

    ani = FuncAnimation(fig, draw, frames=frames, blit=False)
    fig.suptitle("Gradient Descent trên hồi quy tuyến tính", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, .94])
    save_gif(ani, "anim_gradient_descent.gif", fps=10)


if __name__ == "__main__":
    anim_gradient_descent()
    print("Xong.")
