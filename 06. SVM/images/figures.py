"""
Sinh toàn bộ hình minh hoạ cho Lab 06 - Support Vector Machine.

Chạy:  python figures.py
Kết quả: các file .png trong cùng thư mục, được nhúng vào notebook bằng markdown.
Sinh viên có thể sửa script này để tự thí nghiệm với tham số (C, gamma, kernel...).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_blobs, make_moons, make_circles

OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 110, "savefig.bbox": "tight",
    "font.size": 10, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})
C1, C2, C3, C4 = "#2563eb", "#dc2626", "#059669", "#d97706"


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p)
    plt.close(fig)
    print("wrote", name)


def _scatter2(ax, X, y, s=34, alpha=.85, labels=("lớp $-1$", "lớp $+1$")):
    """Vẽ 2 lớp bằng 2 màu/2 marker cố định (xanh = -1, đỏ = +1)."""
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=s, c=C1, marker="o",
               edgecolor="k", linewidth=.5, alpha=alpha, label=labels[0], zorder=3)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=s, c=C2, marker="s",
               edgecolor="k", linewidth=.5, alpha=alpha, label=labels[1], zorder=3)


def _boundary(ax, clf, X, pad=1.0, n=300, fill=True, levels=(-1, 0, 1)):
    """Vẽ decision_function: nền tô nhạt + đường mức -1/0/+1."""
    x0 = np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, n)
    x1 = np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, n)
    XX, YY = np.meshgrid(x0, x1)
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape)
    if fill:
        ax.contourf(XX, YY, np.sign(Z), levels=[-2, 0, 2],
                    colors=[C1, C2], alpha=.10, zorder=0)
    ax.contour(XX, YY, Z, levels=list(levels),
               colors=["k", "k", "k"] if len(levels) == 3 else ["k"],
               linestyles=["--", "-", "--"] if len(levels) == 3 else ["-"],
               linewidths=[1.2, 2.0, 1.2] if len(levels) == 3 else [2.0], zorder=2)
    return XX, YY, Z


# ---------------------------------------------------------------- 1
def fig_margin_geometry():
    """Giải phẫu hình học của margin: siêu phẳng, hai lề, vector w, độ rộng 2/||w||."""
    X, y = make_blobs(n_samples=40, centers=[(1.6, 1.6), (4.4, 4.6)],
                      cluster_std=.72, random_state=7)
    clf = SVC(kernel="linear", C=1000).fit(X, y)
    w, b = clf.coef_[0], clf.intercept_[0]
    nw = np.linalg.norm(w)

    fig, ax = plt.subplots(figsize=(10.0, 7.0))
    _scatter2(ax, X, y)

    xs = np.linspace(-0.4, 8.0, 60)
    line = lambda c: (-w[0] * xs - b + c) / w[1]
    ax.plot(xs, line(0), color="k", lw=2.4, zorder=2,
            label=r"siêu phẳng $w^Tx+b=0$")
    ax.plot(xs, line(1), color=C2, lw=1.5, ls="--", zorder=2,
            label=r"lề dương $w^Tx+b=+1$")
    ax.plot(xs, line(-1), color=C1, lw=1.5, ls="--", zorder=2,
            label=r"lề âm $w^Tx+b=-1$")

    # khoanh tròn support vector
    sv = clf.support_vectors_
    ax.scatter(sv[:, 0], sv[:, 1], s=280, facecolors="none",
               edgecolors=C3, linewidths=2.4, zorder=4,
               label=f"{len(sv)} support vector (chạm lề)")

    # vector w vuông góc với siêu phẳng (vẽ ở phần trên-trái, vùng trống)
    u = w / nw
    pw = np.array([1.15, (-w[0] * 1.15 - b) / w[1]])
    ax.annotate("", xy=pw + u * 1.0, xytext=pw,
                arrowprops=dict(arrowstyle="-|>", color=C4, lw=2.6, mutation_scale=18),
                zorder=5)
    ax.annotate("$w$ = vector pháp tuyến\n(vuông góc với siêu phẳng)",
                xy=tuple(pw + u * 1.05), xytext=(-0.25, 8.45),
                color=C4, fontsize=9.5, fontweight="bold", va="top", ha="left",
                arrowprops=dict(arrowstyle="->", color=C4, lw=1.3),
                bbox=dict(boxstyle="round,pad=.3", fc="white", ec=C4, alpha=.95))

    # mũi tên 2 chiều đo độ rộng margin (vẽ ở phần dưới-phải, vùng trống)
    pm = np.array([4.15, (-w[0] * 4.15 - b) / w[1]])
    q0, q1 = pm - u / nw, pm + u / nw
    ax.annotate("", xy=q1, xytext=q0,
                arrowprops=dict(arrowstyle="<|-|>", color=C3, lw=2.4, mutation_scale=15),
                zorder=5)
    ax.annotate(r"độ rộng margin $=\dfrac{2}{\|w\|}=%.2f$" % (2 / nw),
                xy=tuple(pm), xytext=(6.05, 1.05), color=C3, fontsize=11,
                fontweight="bold", arrowprops=dict(arrowstyle="->", color=C3),
                bbox=dict(boxstyle="round,pad=.35", fc="#ecfdf5", ec=C3))

    ax.text(6.15, 6.10,
            "Dải giữa hai đường đứt nét\nKHÔNG chứa điểm dữ liệu nào\n"
            r"$\Rightarrow y_i(w^Tx_i+b)\geq 1\;\forall i$"
            "\n\nChỉ 3 điểm chạm lề là quan trọng;\nxoá mọi điểm còn lại vẫn ra\nĐÚNG siêu phẳng này.",
            fontsize=9, color="dimgray", va="top",
            bbox=dict(boxstyle="round,pad=.45", fc="#f8f8f8", ec="lightgray"))

    ax.set_xlim(-0.4, 9.4); ax.set_ylim(-0.5, 8.6)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    # legend đặt NGOÀI vùng vẽ (phía dưới) để không che support vector / đường lề
    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.09),
              ncol=3, frameon=False, handlelength=1.9, columnspacing=1.5)
    ax.set_title("Giải phẫu SVM: siêu phẳng, hai lề và độ rộng margin",
                 fontweight="bold")
    save(fig, "01_hinh_hoc_margin.png")


# ---------------------------------------------------------------- 2
def fig_why_large_margin():
    """Vì sao margin lớn tổng quát hoá tốt hơn: một điểm test mới lật kết quả."""
    X, y = make_blobs(n_samples=40, centers=[(1.8, 2.0), (4.6, 4.6)],
                      cluster_std=.60, random_state=3)
    clf = SVC(kernel="linear", C=1000).fit(X, y)
    w, b = clf.coef_[0], clf.intercept_[0]
    nw = np.linalg.norm(w)
    xs = np.linspace(0, 8.4, 60)
    lvl = lambda c: (-w[0] * xs - b + c) / w[1]

    # ba đường tách khác nhau, TẤT CẢ đều chia đúng 100% dữ liệu train
    others = [((-0.30, 4.15), "đường A (thoai thoải)", "#7c3aed"),
              ((-3.50, 13.8), "đường B (dốc đứng)", "#0891b2")]
    # đường C: song song với SVM nhưng ép sát lớp +1
    cC = 0.90

    # điểm test mới: nằm giữa đường C và siêu phẳng SVM  (f = 0.45 > 0 → thật sự là lớp +1)
    tx = 3.05
    test = np.array([tx, (0.45 - b - w[0] * tx) / w[1]])

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.6))

    ax = axes[0]
    _scatter2(ax, X, y)
    for (a, c), lab, col in others:
        ax.plot(xs, a * xs + c, lw=2, color=col, label=lab)
    ax.plot(xs, lvl(cC), lw=2, color=C4, label="đường C (ép sát lớp $+1$)")
    ax.plot(xs, lvl(0), lw=2.8, color="k", label="đường max-margin (SVM)")
    ax.set_title("Có VÔ SỐ đường tách đúng 100% dữ liệu train\n"
                 "→ dựa vào đâu để chọn một đường?", fontsize=10.5)
    ax.set_xlim(0, 8.4); ax.set_ylim(0, 9.4)
    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.11),
              ncol=3, frameon=False, handlelength=1.9, columnspacing=1.3)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")

    ax = axes[1]
    ax.fill_between(xs, lvl(-1), lvl(1), color="k", alpha=.08, zorder=0,
                    label="dải margin của SVM")
    _scatter2(ax, X, y)
    for c, ls in [(1, "--"), (-1, "--")]:
        ax.plot(xs, lvl(c), color="k", lw=1, ls=ls, alpha=.6)
    ax.plot(xs, lvl(cC), lw=2, color=C4, label="đường C: margin HẸP")
    ax.plot(xs, lvl(0), lw=2.8, color="k", label=f"SVM: margin RỘNG ({2/nw:.2f})")
    ax.scatter(*test, marker="*", s=460, color=C2, edgecolor="k",
               linewidth=1.1, zorder=6)
    ax.annotate("Ngôi sao = điểm TEST mới (thật sự là lớp $+1$)\n"
                "• đường C  → đoán lớp $-1$  → SAI\n"
                "• SVM      → đoán lớp $+1$  → ĐÚNG",
                xy=tuple(test), xytext=(0.15, 9.25), fontsize=9.5,
                va="top", ha="left",
                arrowprops=dict(arrowstyle="->", color="dimgray"),
                bbox=dict(boxstyle="round,pad=.45", fc="#fff7ed", ec=C4))
    ax.set_title("Margin rộng = vùng đệm an toàn\n"
                 "dữ liệu mới lệch một chút vẫn không bị lật nhãn", fontsize=10.5)
    ax.set_xlim(0, 8.4); ax.set_ylim(0, 9.4)
    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.11),
              ncol=2, frameon=False, handlelength=1.9, columnspacing=1.3)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")

    fig.suptitle("Vì sao SVM chọn đường có margin LỚN NHẤT", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "02_vi_sao_margin_lon.png")


# ---------------------------------------------------------------- 3
def fig_hard_vs_soft():
    """Một outlier duy nhất đủ bóp méo hard-margin; soft-margin thì bỏ qua được."""
    X, y = make_blobs(n_samples=40, centers=[(1.8, 2.0), (4.6, 4.6)],
                      cluster_std=.60, random_state=3)
    Xo = np.vstack([X, [[2.55, 2.75]]])       # outlier lớp đỏ lọt sâu vào lớp xanh
    yo = np.r_[y, 1]

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.0))
    xs = np.linspace(0, 6.6, 50)

    def draw(ax, Xa, ya, C, title):
        clf = SVC(kernel="linear", C=C).fit(Xa, ya)
        w, b = clf.coef_[0], clf.intercept_[0]
        _scatter2(ax, Xa, ya)
        ax.plot(xs, (-w[0] * xs - b) / w[1], color="k", lw=2.4)
        for s in (1, -1):
            ax.plot(xs, (-w[0] * xs - b + s) / w[1], color="k", lw=1, ls="--", alpha=.7)
        ax.fill_between(xs, (-w[0] * xs - b - 1) / w[1], (-w[0] * xs - b + 1) / w[1],
                        color="k", alpha=.07, zorder=0)
        ax.set_xlim(0, 6.6); ax.set_ylim(0, 6.6)
        ax.set_title(title + f"\nmargin $=2/\\|w\\|=${2/np.linalg.norm(w):.2f}", fontsize=10)
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
        return clf, w, b

    draw(axes[0], X, y, 1e6, "Dữ liệu sạch — hard-margin ($C\\to\\infty$)")

    clf, w, b = draw(axes[1], Xo, yo, 1e6,
                     "THÊM 1 OUTLIER — hard-margin bắt buộc\nphải chiều nó → margin co lại gần bằng 0")
    axes[1].scatter(*Xo[-1], marker="*", s=430, color=C2, edgecolor="k",
                    linewidth=1.1, zorder=6)
    axes[1].annotate("outlier", xy=Xo[-1], xytext=(4.55, 2.15), fontsize=9.5,
                     color=C2, fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color=C2))

    clf, w, b = draw(axes[2], Xo, yo, 0.1,
                     "Soft-margin ($C=0.1$) — chấp nhận outlier vi phạm\n"
                     "để giữ margin rộng, tổng quát hoá tốt hơn")
    axes[2].scatter(*Xo[-1], marker="*", s=430, color=C2, edgecolor="k",
                    linewidth=1.1, zorder=6)
    # mũi tên slack xi_i: từ điểm tới lề của lớp nó
    nw2 = w @ w
    for xi, yi in zip(Xo, yo):
        s = 1 if yi == 1 else -1
        f = w @ xi + b
        slack = max(0.0, 1 - s * f)
        if slack > 1e-3:
            target = xi + w * (s * 1 - f) / nw2      # chiếu lên lề y=±1
            axes[2].annotate("", xy=target, xytext=xi,
                             arrowprops=dict(arrowstyle="-|>", color=C4, lw=1.8,
                                             mutation_scale=12), zorder=5)
    axes[2].plot([], [], color=C4, lw=1.8,
                 label=r"$\xi_i$ = mức vi phạm lề")
    # legend + công thức đặt DƯỚI panel để không đè lên điểm dữ liệu / đường lề
    axes[2].legend(fontsize=8.5, loc="upper right", bbox_to_anchor=(1.0, -0.15),
                   frameon=False)
    axes[2].text(0.0, -0.26,
                 r"$y_i(w^Tx_i+b)\geq 1-\xi_i,\;\xi_i\geq 0$" "\n"
                 r"phạt $C\sum_i \xi_i$ trong hàm mục tiêu",
                 transform=axes[2].transAxes, va="top", ha="left", fontsize=9,
                 bbox=dict(boxstyle="round,pad=.35", fc="#fff7ed", ec=C4))

    fig.suptitle("Hard-margin giòn như thuỷ tinh — soft-margin mới dùng được ngoài đời",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0.10, 1, 0.94])
    save(fig, "03_hard_vs_soft_margin.png")


# ---------------------------------------------------------------- 4
def fig_effect_of_C():
    """C điều khiển đánh đổi: margin rộng & nhiều SV  <->  bám sát dữ liệu."""
    X, y = make_blobs(n_samples=100, centers=[(0, 0), (4, 4)],
                      cluster_std=2.2, random_state=1)   # hai lớp CÓ chồng lấn
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.2))
    for ax, C in zip(axes, [0.01, 1, 100]):
        clf = SVC(kernel="linear", C=C).fit(X, y)
        _boundary(ax, clf, X, pad=2.8)      # chừa lề rộng để chú thích không đè điểm
        _scatter2(ax, X, y)
        sv = clf.support_vectors_
        ax.scatter(sv[:, 0], sv[:, 1], s=180, facecolors="none",
                   edgecolors=C3, linewidths=1.8, zorder=4)
        ax.set_title(f"C = {C}\n{len(sv)} support vector, "
                     f"margin = {2/np.linalg.norm(clf.coef_):.2f}, "
                     f"train acc = {clf.score(X, y)*100:.1f}%", fontsize=10)
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    axes[0].legend(fontsize=8, loc="upper left", framealpha=.95)
    # chú thích đặt DƯỚI mỗi panel: bên trong panel sẽ đè lên support vector
    box = dict(boxstyle="round,pad=.35", fc="white", ec="lightgray", alpha=.95)
    axes[0].annotate("C nhỏ → khoan dung với vi phạm\nmargin RỘNG, NHIỀU support vector\n"
                 "(bias cao, variance thấp)",
                 xy=(0, 0), xycoords="axes fraction", textcoords="offset points",
                 xytext=(0, -42), fontsize=8.5, color="#444444",
                 va="top", ha="left", bbox=box)
    axes[1].annotate("C vừa → cân bằng\n(mặc định của sklearn là C = 1)",
                 xy=(0, 0), xycoords="axes fraction", textcoords="offset points",
                 xytext=(0, -42), fontsize=8.5, color="#444444",
                 va="top", ha="left", bbox=box)
    axes[2].annotate("C lớn → khắt khe, cố gò sát dữ liệu\nmargin HẸP, ÍT support vector\n"
                 "(bias thấp, variance cao → dễ overfit)",
                 xy=(0, 0), xycoords="axes fraction", textcoords="offset points",
                 xytext=(0, -42), fontsize=8.5, color="#444444",
                 va="top", ha="left", bbox=box)
    fig.suptitle(r"Tham số $C$ chính là nghịch đảo của mức regularization ($C \approx 1/\lambda$)",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    save(fig, "04_anh_huong_tham_so_C.png")


# ---------------------------------------------------------------- 5
def fig_kernel_trick_3d():
    """Hình quan trọng nhất của bài: nâng chiều biến bài toán cong thành phẳng."""
    X, y = make_circles(n_samples=180, factor=.42, noise=.09, random_state=1)
    z = X[:, 0] ** 2 + X[:, 1] ** 2
    z0 = (z[y == 0].min() + z[y == 1].max()) / 2     # ngưỡng tách trên trục thứ 3

    fig = plt.figure(figsize=(14.5, 6.2))

    ax = fig.add_axes([0.05, 0.08, 0.34, 0.74])
    _scatter2(ax, X, y, s=26, labels=("lớp ngoài", "lớp trong"))
    th = np.linspace(0, 2 * np.pi, 300)
    r = np.sqrt(z0)
    ax.plot(r * np.cos(th), r * np.sin(th), color="k", lw=2.2,
            label="ranh giới đúng: một ĐƯỜNG TRÒN")
    ax.set_aspect("equal")
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.95)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.legend(fontsize=8.5, loc="upper center")
    ax.set_title("KHÔNG GIAN GỐC 2D\nkhông một ĐƯỜNG THẲNG nào tách nổi", fontsize=11)

    # mũi tên "nâng chiều" ở giữa hai panel
    fig.text(.400, .60, r"$\phi$", fontsize=22, color=C4, fontweight="bold",
             ha="center", va="center")
    ax.annotate("", xy=(.444, .52), xytext=(.356, .52), xycoords="figure fraction",
                arrowprops=dict(arrowstyle="-|>", color=C4, lw=3, mutation_scale=22))
    fig.text(.400, .455, "nâng chiều", fontsize=9.5, color=C4, ha="center", va="top")

    ax = fig.add_axes([0.455, 0.05, 0.415, 0.72], projection="3d")
    ax.scatter(X[y == 0, 0], X[y == 0, 1], z[y == 0], s=20, c=C1, depthshade=False,
               edgecolor="k", linewidth=.3)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], z[y == 1], s=20, c=C2, marker="s",
               depthshade=False, edgecolor="k", linewidth=.3)
    g = np.linspace(-1.3, 1.3, 12)
    G1, G2 = np.meshgrid(g, g)
    ax.plot_surface(G1, G2, np.full_like(G1, z0), alpha=.40, color=C3,
                    linewidth=0, antialiased=True)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.set_zlabel("$x_3 = x_1^2 + x_2^2$")
    ax.view_init(elev=16, azim=-62)
    ax.set_title(r"SAU KHI NÂNG CHIỀU $\phi(x)=(x_1,\,x_2,\,x_1^2+x_2^2)$"
                 "\nmột MẶT PHẲNG $x_3 = %.2f$ tách được hoàn hảo" % z0, fontsize=11)

    fig.suptitle("Kernel trick: bài toán khó ở chiều thấp có thể trở thành dễ ở chiều cao\n"
                 r"và SVM không cần tính $\phi(x)$ — chỉ cần $K(x,x')=\langle\phi(x),\phi(x')\rangle$",
                 fontweight="bold", fontsize=11.5)
    save(fig, "05_kernel_trick_nang_chieu_3d.png")


# ---------------------------------------------------------------- 6
def fig_gamma_rbf():
    """gamma nhỏ = ảnh hưởng lan xa (mượt); gamma lớn = mỗi điểm một ốc đảo (overfit)."""
    X, y = make_moons(n_samples=220, noise=.22, random_state=42)
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.3))
    notes = ["quá mượt — gần như tuyến tính\n(UNDERFIT)",
             "vừa đẹp — bám hình lưỡi liềm",
             "bắt đầu uốn éo theo nhiễu",
             "mỗi điểm một 'ốc đảo' riêng\n(OVERFIT nặng)"]
    for ax, g, note in zip(axes, [0.1, 1, 10, 100], notes):
        clf = SVC(kernel="rbf", C=1.0, gamma=g).fit(X, y)
        _boundary(ax, clf, X, pad=.5, levels=(0,))
        _scatter2(ax, X, y, s=20)
        ax.set_title(f"gamma = {g}\ntrain acc = {clf.score(X, y)*100:.1f}%, "
                     f"{len(clf.support_vectors_)} SV", fontsize=10)
        # chú thích đặt DƯỚI panel: bên trong panel sẽ đè lên đường ranh giới
        ax.annotate(note, xy=(0, 0), xycoords="axes fraction",
                    textcoords="offset points", xytext=(0, -42), fontsize=8.5,
                    color="#444444", va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=.3", fc="white", ec="lightgray", alpha=.95))
        ax.set_xlabel("$x_1$")
    axes[0].set_ylabel("$x_2$")
    axes[0].legend(fontsize=8, loc="upper left", framealpha=.95)
    fig.suptitle(r"RBF kernel: $\gamma$ là NGHỊCH ĐẢO bề rộng ảnh hưởng của mỗi support vector",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    save(fig, "06_anh_huong_gamma_rbf.png")


# ---------------------------------------------------------------- 7
def fig_compare_kernels():
    """Cùng một dữ liệu, 5 kernel cho 5 hình dạng ranh giới rất khác nhau."""
    X, y = make_moons(n_samples=220, noise=.22, random_state=0)
    Xs = (X - X.mean(0)) / X.std(0)     # kernel poly/sigmoid rất nhạy scale
    cfg = [(dict(kernel="linear"), "linear\n$K=x^Tx'$"),
           (dict(kernel="poly", degree=2, coef0=1), "poly bậc 2\n$K=(x^Tx'+1)^2$"),
           (dict(kernel="poly", degree=3, coef0=1), "poly bậc 3\n$K=(x^Tx'+1)^3$"),
           (dict(kernel="rbf", gamma="scale"), "RBF\n$K=e^{-\\gamma\\|x-x'\\|^2}$"),
           (dict(kernel="sigmoid", gamma="scale", coef0=0), "sigmoid\n$K=\\tanh(\\gamma x^Tx'+c)$")]
    fig, axes = plt.subplots(1, 5, figsize=(19.5, 4.1))
    for ax, (kw, name) in zip(axes, cfg):
        clf = SVC(C=1.0, **kw).fit(Xs, y)
        _boundary(ax, clf, Xs, pad=.4, levels=(0,))
        _scatter2(ax, Xs, y, s=18)
        ax.set_title(f"{name}\ntrain acc = {clf.score(Xs, y)*100:.1f}%", fontsize=9.5)
        ax.set_xlabel("$x_1$ (đã scale)")
    axes[0].set_ylabel("$x_2$ (đã scale)")
    # một legend chung đặt NGOÀI panel: legend trong panel 1 che mất ranh giới
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=9.5, frameon=False,
               bbox_to_anchor=(0.5, 0.005), handlelength=1.6, columnspacing=2.0)
    fig.suptitle("Cùng dữ liệu — mỗi kernel là một 'giả thuyết' khác nhau về hình dạng ranh giới\n"
                 "(sigmoid hiếm khi thắng RBF, chủ yếu còn tồn tại vì lý do lịch sử)",
                 fontweight="bold", fontsize=11)
    fig.tight_layout(rect=[0, 0.07, 1, 0.93])
    save(fig, "07_so_sanh_cac_kernel.png")


# ---------------------------------------------------------------- 8
def fig_hinge_loss():
    """Hinge loss = cận trên lồi của 0-1 loss; so với log loss của Logistic Regression."""
    m = np.linspace(-3, 3, 600)          # m = y * f(x), gọi là functional margin
    zero_one = (m < 0).astype(float)
    hinge = np.maximum(0, 1 - m)
    sq_hinge = np.maximum(0, 1 - m) ** 2
    logistic = np.log2(1 + np.exp(-m))   # chia log(2) để chạm (0, 1) cho dễ so sánh

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8))
    ax = axes[0]
    ax.step(m, zero_one, where="post", color="k", lw=2.4,
            label="0-1 loss (mục tiêu thật sự)")
    ax.plot(m, hinge, color=C2, lw=2.4, label=r"hinge $\max(0,\,1-m)$ — SVM")
    ax.plot(m, sq_hinge, color=C4, lw=2, ls="--",
            label=r"squared hinge $\max(0,\,1-m)^2$")
    ax.plot(m, logistic, color=C1, lw=2,
            label=r"log loss $\log_2(1+e^{-m})$ — LogReg")
    ax.axvline(0, color="gray", lw=1)
    ax.plot([1, 1], [-.15, 2.30], color=C3, lw=1.2, ls=":")
    ax.text(1.12, 1.42, "$m=1$: mép lề", color=C3, fontsize=9,
            bbox=dict(boxstyle="round,pad=.2", fc="white", ec="none", alpha=.85))
    ax.set_xlabel(r"functional margin $m = y\,(w^Tx+b)$")
    ax.set_ylabel("mất mát")
    ax.set_ylim(-.15, 3.2)
    ax.legend(fontsize=8.5, loc="upper right", framealpha=.95)
    ax.set_title("Hinge loss là CẬN TRÊN LỒI của 0-1 loss\n"
                 "→ tối thiểu hinge thì cũng ép 0-1 loss xuống, mà lại tối ưu được", fontsize=10)

    ax = axes[1]
    ax.plot(m, hinge, color=C2, lw=2.6, label="hinge (SVM)")
    ax.plot(m, logistic, color=C1, lw=2.2, label="log loss (Logistic Regression)")
    ax.fill_between(m, 0, hinge, where=(m >= 1), color=C3, alpha=.25)
    ax.plot([1, 1], [-.15, 1.35], color=C3, lw=1.2, ls=":")
    ax.annotate("Hinge = 0 HOÀN TOÀN khi $m\\geq1$\n"
                "→ điểm ngoài lề KHÔNG đóng góp gradient\n"
                "→ nghiệm chỉ phụ thuộc support vector\n"
                "    (mô hình THƯA)",
                xy=(1.9, .02), xytext=(.18, 2.15), fontsize=9,
                va="top", ha="left",
                arrowprops=dict(arrowstyle="->", color=C3),
                bbox=dict(boxstyle="round,pad=.4", fc="#ecfdf5", ec=C3))
    ax.annotate("Log loss > 0 với MỌI điểm\n→ mọi mẫu đều kéo nghiệm một chút\n(mô hình KHÔNG thưa)",
                xy=(2.4, logistic[np.argmin(np.abs(m - 2.4))]), xytext=(.18, 3.14),
                fontsize=9, va="top", ha="left",
                arrowprops=dict(arrowstyle="->", color=C1),
                bbox=dict(boxstyle="round,pad=.4", fc="#eff6ff", ec=C1))
    ax.set_xlabel(r"$m = y\,(w^Tx+b)$"); ax.set_ylabel("mất mát")
    ax.set_ylim(-.15, 3.2)
    ax.legend(fontsize=9, loc="lower left", framealpha=.95)
    ax.set_title("SVM và Logistic Regression khác nhau ở ĐÚNG một chỗ: hàm mất mát", fontsize=10)

    fig.suptitle("Đọc SVM dưới lăng kính hàm mất mát: "
                 r"$\min_w \frac{1}{2}\|w\|^2 + C\sum_i \max(0,\,1-y_i(w^Tx_i+b))$",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    save(fig, "08_hinge_loss_vs_cac_loss_khac.png")


# ---------------------------------------------------------------- 9
def fig_rbf_similarity():
    """RBF kernel = 'độ giống nhau' giảm dần theo khoảng cách, gamma quyết định tốc độ giảm."""
    d = np.linspace(0, 4, 500)
    fig = plt.figure(figsize=(14.0, 4.9))

    ax = fig.add_axes([0.055, 0.14, 0.39, 0.68])
    for g, col in zip([0.1, 1, 10, 100], [C1, C3, C4, C2]):
        ax.plot(d, np.exp(-g * d ** 2), lw=2.2, color=col, label=f"$\\gamma$ = {g}")
    ax.axhline(.5, color="gray", ls=":", lw=1)
    ax.text(3.97, .545, "mức 'giống nhau một nửa'", fontsize=8.5, color="#444444",
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=.22", fc="white", ec="none", alpha=.88))
    ax.set_xlabel(r"khoảng cách $\|x-x'\|$")
    ax.set_ylabel(r"$K(x,x') = \exp(-\gamma\|x-x'\|^2)$")
    ax.legend(fontsize=9)
    ax.set_title(r"$\gamma$ lớn $\Rightarrow$ độ giống nhau tụt về 0 rất nhanh"
                 "\n$\\Rightarrow$ mỗi điểm chỉ 'nhìn thấy' hàng xóm cực gần", fontsize=10)

    # bản đồ vùng ảnh hưởng 2D của 3 support vector
    svs = np.array([[-1.4, -1.0], [1.5, .3], [0.0, 1.7]])
    gx = np.linspace(-3, 3, 240)
    GX, GY = np.meshgrid(gx, gx)
    for k, g in enumerate([0.3, 3, 30]):
        a = fig.add_axes([0.525 + k * 0.158, 0.16, 0.145, 0.60])
        S = sum(np.exp(-g * ((GX - s[0]) ** 2 + (GY - s[1]) ** 2)) for s in svs)
        a.contourf(GX, GY, S, levels=20, cmap="YlOrRd", vmin=0, vmax=1)
        a.scatter(svs[:, 0], svs[:, 1], marker="x", s=80, c="k", linewidths=2.2)
        a.set_title(f"$\\gamma$ = {g}", fontsize=10)
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
        a.set_aspect("equal")
    fig.text(0.762, 0.855, "Vùng ảnh hưởng của 3 support vector (dấu ×)",
             ha="center", fontsize=10)
    fig.text(0.762, 0.055, "càng đỏ = kernel càng lớn = càng 'giống' một support vector\n"
             r"$\gamma$ lớn → các ốc đảo tách rời → model chỉ nhớ từng điểm một",
             ha="center", fontsize=9, color="#444444")

    fig.suptitle(r"RBF kernel đo ĐỘ GIỐNG NHAU theo khoảng cách — $\gamma$ = nghịch đảo bề rộng ảnh hưởng",
                 fontweight="bold")
    save(fig, "09_rbf_kernel_do_giong_nhau.png")


# ---------------------------------------------------------------- 10
def fig_svr_epsilon_tube():
    """Giới thiệu SVR: thay vì lề rỗng, ta muốn dữ liệu NẰM TRONG ống epsilon."""
    from sklearn.svm import SVR
    rng = np.random.default_rng(4)
    x = np.sort(rng.uniform(0, 10, 70))
    y = np.sin(x) * 2 + .25 * x + rng.normal(0, .35, 70)
    eps = 0.6
    m = SVR(kernel="rbf", C=10, gamma=.3, epsilon=eps).fit(x.reshape(-1, 1), y)
    xs = np.linspace(0, 10, 400)
    yh = m.predict(xs.reshape(-1, 1))

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.fill_between(xs, yh - eps, yh + eps, color=C3, alpha=.22,
                    label=rf"ống $\epsilon$-insensitive ($\epsilon={eps}$)")
    ax.plot(xs, yh, color=C3, lw=2.4, label="SVR")
    inside = np.abs(y - m.predict(x.reshape(-1, 1))) <= eps + 1e-9
    ax.scatter(x[inside], y[inside], s=28, color=C1, edgecolor="k", linewidth=.4,
               label="điểm TRONG ống → mất mát = 0")
    ax.scatter(x[~inside], y[~inside], s=52, color=C2, edgecolor="k", linewidth=.5,
               marker="s", label="điểm NGOÀI ống → support vector, bị phạt")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.95)          # chừa chỗ cho legend, tránh đè lên dữ liệu
    ax.legend(fontsize=8.5, loc="upper left", framealpha=.95)
    ax.set_title("SVR — Support Vector Regression\n"
                 r"mất mát $= \max(0,\,|y-\hat{y}|-\epsilon)$: sai lệch nhỏ hơn $\epsilon$ được tha bổng",
                 fontsize=10.5, fontweight="bold")
    save(fig, "10_svr_ong_epsilon.png")


if __name__ == "__main__":
    fig_margin_geometry()
    fig_why_large_margin()
    fig_hard_vs_soft()
    fig_effect_of_C()
    fig_kernel_trick_3d()
    fig_gamma_rbf()
    fig_compare_kernels()
    fig_hinge_loss()
    fig_rbf_similarity()
    fig_svr_epsilon_tube()
    print("Xong.")
