"""
Sinh toàn bộ hình minh hoạ cho Lab 04 - K-Nearest Neighbors (KNN).

Chạy:  python figures.py
Kết quả: các file .png trong cùng thư mục, được nhúng vào notebook bằng markdown.
Sinh viên có thể sửa script này để tự thí nghiệm với tham số (đổi k, đổi metric,
đổi số chiều...) rồi chạy lại để xem hình thay đổi thế nào.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import ListedColormap

OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 110, "savefig.bbox": "tight",
    "font.size": 10, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})
C1, C2, C3, C4 = "#2563eb", "#dc2626", "#059669", "#d97706"
CMAP_BG = ListedColormap(["#dbeafe", "#fee2e2"])       # nền 2 lớp (nhạt)
CMAP_BG3 = ListedColormap(["#dbeafe", "#fee2e2", "#d1fae5"])


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p)
    plt.close(fig)
    print("wrote", name)


def _moons(n=300, noise=0.28, seed=0):
    from sklearn.datasets import make_moons
    return make_moons(n_samples=n, noise=noise, random_state=seed)


def _boundary(ax, model, X, y, h=0.02, cmap=CMAP_BG, show_pts=True):
    """Vẽ vùng quyết định của model đã fit lên lưới bao quanh X."""
    x0, x1 = X[:, 0].min() - .5, X[:, 0].max() + .5
    y0, y1 = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x0, x1, h), np.arange(y0, y1, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.pcolormesh(xx, yy, Z, cmap=cmap, shading="auto", zorder=0)
    ax.contour(xx, yy, Z, levels=[0.5], colors="k", linewidths=1.1, zorder=1)
    if show_pts:
        ax.scatter(X[y == 0, 0], X[y == 0, 1], s=16, color=C1, edgecolor="w",
                   linewidth=.3, zorder=3)
        ax.scatter(X[y == 1, 0], X[y == 1, 1], s=16, color=C2, marker="s",
                   edgecolor="w", linewidth=.3, zorder=3)
    ax.set_xlim(x0, x1 - h); ax.set_ylim(y0, y1 - h)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)


# ---------------------------------------------------------------- 1
def fig_how_knn_works():
    """Đổi k → đổi kết quả biểu quyết, dù điểm truy vấn không hề nhúc nhích."""
    rng = np.random.default_rng(11)
    q = np.array([2.40, 2.20])
    blue = rng.normal([2.0, 2.0], 0.85, size=(34, 2))
    # dọn một khoảng trống nhỏ quanh q để hình vẽ dễ đọc
    blue = blue[np.linalg.norm(blue - q, axis=1) > 0.60]
    red = rng.normal([4.9, 4.6], 0.70, size=(22, 2))
    red = np.vstack([red, q + [0.30, 0.31]])      # một điểm ĐỎ lạc vào vùng XANH

    X = np.vstack([blue, red])
    y = np.r_[np.zeros(len(blue)), np.ones(len(red))]
    d = np.linalg.norm(X - q, axis=1)
    order = np.argsort(d)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, k in zip(axes, [1, 5]):
        nn = order[:k]
        r = d[nn].max() * 1.06
        ax.add_patch(Circle(q, r, fill=True, color=C4, alpha=.10, zorder=0))
        ax.add_patch(Circle(q, r, fill=False, color=C4, lw=1.8, ls="--", zorder=2))
        ax.scatter(blue[:, 0], blue[:, 1], s=42, color=C1, alpha=.85, label="lớp XANH")
        ax.scatter(red[:, 0], red[:, 1], s=42, color=C2, marker="s", alpha=.85,
                   label="lớp ĐỎ")
        for i in nn:
            ax.plot([q[0], X[i, 0]], [q[1], X[i, 1]], color="gray", lw=1.1,
                    ls=":", zorder=1)
        ax.scatter(*q, marker="*", s=280, color=C4, edgecolor="k", linewidth=.8,
                   zorder=5, label="điểm cần dự đoán")
        n_red = int(y[nn].sum()); n_blue = k - n_red
        win = "ĐỎ" if n_red > n_blue else "XANH"
        col = C2 if n_red > n_blue else C1
        ax.set_title(f"k = {k}:  {n_blue} phiếu XANH  vs  {n_red} phiếu ĐỎ"
                     f"  →  dự đoán {win}", fontsize=10.5, color=col)
        ax.set_xlim(-0.2, 7.0); ax.set_ylim(-0.3, 6.9)
        ax.set_xlabel("feature 1"); ax.set_ylabel("feature 2")
        ax.set_aspect("equal")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[1].text(3.15, 0.05, "Cùng một điểm, chỉ đổi k\n→ ĐỔI LUÔN KẾT QUẢ.\n"
                             "k không phải tham số 'chọn đại'.",
                 fontsize=9, color="dimgray",
                 bbox=dict(boxstyle="round", fc="#fff7ed", ec=C4, alpha=.9))
    fig.suptitle("KNN hoạt động thế nào: khoanh vùng k hàng xóm gần nhất rồi biểu quyết",
                 fontweight="bold")
    fig.tight_layout()
    save(fig, "01_cach_knn_hoat_dong.png")


# ---------------------------------------------------------------- 2
def fig_boundary_by_k():
    from sklearn.neighbors import KNeighborsClassifier
    X, y = _moons(300, 0.22, 0)
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.1))
    notes = ["k = 1 — mỗi điểm nhiễu tự tạo một 'ốc đảo'\nOVERFIT (variance cao)",
             "k = 5 — bắt được hình dạng hai vầng trăng\nVỪA ĐẸP",
             "k = 15 — mượt hơn, bỏ qua nhiễu lẻ\nvẫn tốt",
             "k = 50 — quá mượt, nuốt luôn chi tiết\nUNDERFIT (bias cao)"]
    for ax, k, note in zip(axes, [1, 5, 15, 50], notes):
        m = KNeighborsClassifier(n_neighbors=k).fit(X, y)
        _boundary(ax, m, X, y)
        ax.set_title(note, fontsize=9.5)
    fig.suptitle("k điều khiển ĐỘ PHỨC TẠP của KNN: k nhỏ = ranh giới lởm chởm, "
                 "k lớn = ranh giới mượt", fontweight="bold")
    fig.tight_layout()
    save(fig, "02_bien_quyet_dinh_theo_k.png")


# ---------------------------------------------------------------- 3
def fig_bias_variance_k():
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    X, y = _moons(400, 0.30, 1)
    ks = np.arange(1, 151)
    TR = np.zeros((8, len(ks))); TE = np.zeros((8, len(ks)))
    for s_ in range(8):                       # trung bình 8 lần chia để đường mượt
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.5, random_state=s_,
                                              stratify=y)
        for j, k in enumerate(ks):
            m = KNeighborsClassifier(n_neighbors=int(k)).fit(Xtr, ytr)
            TR[s_, j] = m.score(Xtr, ytr); TE[s_, j] = m.score(Xte, yte)
    tr, te = TR.mean(0) * 100, TE.mean(0) * 100
    best = int(ks[int(np.argmax(te))])
    sqrtN = np.sqrt(200)

    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    ax.plot(ks, tr, color=C1, lw=2.2, label="Train accuracy")
    ax.plot(ks, te, color=C2, lw=2.2, label="Test accuracy")
    ax.fill_between(ks, te, tr, color="#94a3b8", alpha=.20)
    ax.axvline(best, color=C3, ls="--", lw=1.7)
    ax.axvline(sqrtN, color=C4, ls=":", lw=2.0)
    ax.annotate("k = 1: train 100% nhưng test thấp nhất\n"
                "→ VARIANCE cao (học thuộc lòng cả nhiễu)",
                xy=(1, 100), xytext=(24, 100.6), fontsize=9, color="dimgray",
                arrowprops=dict(arrowstyle="->", color="gray"))
    ax.annotate("k quá lớn: train và test cùng tụt\n"
                "→ BIAS cao (mọi điểm gần như bầu chung một kết quả)",
                xy=(146, te[145]), xytext=(46, 76.3), fontsize=9, color="dimgray",
                arrowprops=dict(arrowstyle="->", color="gray"))
    ax.text(20, 93.0,
            f"k tốt nhất (đường xanh lá) = {best}\n"
            + r"$\sqrt{N}$" + f" = √200 ≈ {sqrtN:.0f} (đường cam)\n"
            "→ quy tắc √N rơi đúng vùng tốt ở đây,\n"
            "   nhưng vẫn PHẢI kiểm chứng bằng cross-validation",
            fontsize=8.8, color="#334155",
            bbox=dict(boxstyle="round", fc="#f8fafc", ec="#94a3b8", alpha=.95))
    ax.text(86, 93.2, "vùng xám = khoảng cách train–test\n= mức OVERFIT",
            fontsize=8.6, color="#475569")
    ax.set_xlabel("k   (độ phức tạp mô hình GIẢM dần khi k tăng  →)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(75.5, 103)
    ax.legend(fontsize=9, loc="lower left")
    ax.set_title("Đường cong bias–variance của KNN\n"
                 "k nhỏ = variance cao, k lớn = bias cao, điểm ngọt nằm ở giữa",
                 fontweight="bold", fontsize=11)
    save(fig, "03_bias_variance_theo_k.png")


# ---------------------------------------------------------------- 4
def fig_why_scale():
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(5)
    n = 60
    age = rng.uniform(20, 70, n)
    income = rng.uniform(20_000, 200_000, n)
    # nhãn thật phụ thuộc CHỦ YẾU vào tuổi
    y = (age > 45).astype(int)
    X = np.c_[age, income]
    q = np.array([35.0, 110_000.0])   # tuổi 35 → nhãn ĐÚNG phải là lớp 0

    k = 5
    d_raw = np.linalg.norm(X - q, axis=1)
    nn_raw = np.argsort(d_raw)[:k]

    sc = StandardScaler().fit(X)
    Xs, qs = sc.transform(X), sc.transform(q.reshape(1, -1))[0]
    d_s = np.linalg.norm(Xs - qs, axis=1)
    nn_s = np.argsort(d_s)[:k]

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6))
    for ax, (P, Q, nn, ttl, xl, yl) in zip(axes, [
            (X, q, nn_raw, "TRƯỚC khi scale — Income (đơn vị vạn) nuốt chửng Age",
             "Age (20–70)", "Income (20.000–200.000)"),
            (Xs, qs, nn_s, "SAU StandardScaler — hai trục có tiếng nói ngang nhau",
             "Age (đã chuẩn hoá)", "Income (đã chuẩn hoá)")]):
        ax.scatter(P[y == 0, 0], P[y == 0, 1], s=34, color=C1, alpha=.75, label="lớp 0 (trẻ)")
        ax.scatter(P[y == 1, 0], P[y == 1, 1], s=34, color=C2, marker="s", alpha=.75,
                   label="lớp 1 (lớn tuổi)")
        for i in nn:
            ax.plot([Q[0], P[i, 0]], [Q[1], P[i, 1]], color=C4, lw=1.4, ls="--", zorder=2)
        ax.scatter(P[nn, 0], P[nn, 1], s=150, facecolor="none", edgecolor=C4,
                   linewidth=2, zorder=3, label=f"{k} hàng xóm gần nhất")
        ax.scatter(*Q, marker="*", s=380, color=C4, edgecolor="k", linewidth=.8,
                   zorder=5, label="điểm cần dự đoán")
        n1 = int(y[nn].sum())
        verdict = "SAI" if n1 > k - n1 else "ĐÚNG"
        ax.set_title(f"{ttl}\nBiểu quyết: {k - n1} phiếu lớp 0  vs  {n1} phiếu lớp 1"
                     f"  →  dự đoán {verdict}", fontsize=9.8)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.legend(fontsize=7.5, loc="upper left")
    axes[0].text(20.5, 22_000,
                 "Khoảng cách gần như CHỈ đo chênh lệch Income:\n"
                 "chênh 1 tuổi ≈ 1 đơn vị, chênh 1 nghìn đồng cũng ≈ 1 đơn vị\n"
                 "→ hàng xóm được chọn theo Income, còn Age bị bỏ qua.",
                 fontsize=8.4, color="dimgray",
                 bbox=dict(boxstyle="round", fc="#fef2f2", ec=C2, alpha=.9))
    fig.suptitle("Vì sao BẮT BUỘC phải chuẩn hoá trước khi dùng KNN — "
                 "đổi scale là đổi luôn tập hàng xóm", fontweight="bold")
    fig.tight_layout()
    save(fig, "04_vi_sao_phai_scale.png")


# ---------------------------------------------------------------- 5
def fig_minkowski_balls():
    def ball(p, n=800):
        t = np.linspace(0, np.pi / 2, n)
        # đường |x|^p + |y|^p = 1 trên góc phần tư thứ nhất
        x = np.cos(t) ** (2 / p)
        yv = np.sin(t) ** (2 / p)
        X = np.r_[x, -x[::-1], -x, x[::-1]]
        Y = np.r_[yv, yv[::-1], -yv, -yv[::-1]]
        return X, Y

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.4))

    ax = axes[0]
    for p, name, col in [(1, "L1 — Manhattan (hình thoi)", C1),
                         (2, "L2 — Euclidean (hình tròn)", C3)]:
        X, Y = ball(p)
        ax.plot(X, Y, color=col, lw=2.2, label=name)
    ax.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], color=C2, lw=2.2,
            label=r"L$\infty$ — Chebyshev (hình vuông)")
    ax.set_title("'Quả cầu đơn vị': tập hợp các điểm\ncách gốc đúng 1 đơn vị",
                 fontsize=10)
    ax.legend(fontsize=8, loc="upper center")
    ax.set_aspect("equal"); ax.set_xlim(-1.75, 1.75); ax.set_ylim(-1.45, 2.25)
    ax.axhline(0, color="k", lw=.7); ax.axvline(0, color="k", lw=.7)

    ax = axes[1]
    for p, col in zip([0.5, 1, 1.5, 2, 4, 8], plt.cm.viridis(np.linspace(0, .9, 6))):
        X, Y = ball(p)
        ax.plot(X, Y, color=col, lw=1.9, label=f"p = {p}")
    ax.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], "k--", lw=1.4, label=r"p → $\infty$")
    ax.set_title("Minkowski tổng quát\n"
                 r"$d_p(u,v)=\left(\sum_i |u_i-v_i|^p\right)^{1/p}$", fontsize=10)
    ax.legend(fontsize=7.5, ncol=4, loc="upper center", columnspacing=.9,
              handlelength=1.4)
    ax.set_aspect("equal"); ax.set_xlim(-1.75, 1.75); ax.set_ylim(-1.45, 2.25)
    ax.axhline(0, color="k", lw=.7); ax.axvline(0, color="k", lw=.7)

    # (c) đổi metric → đổi hàng xóm gần nhất
    ax = axes[2]
    q = np.array([0., 0.])
    A = np.array([0.85, 0.0]); B = np.array([0.50, 0.50])
    for p, col, ls, nm in [(1, C1, "-", "vòng L1 qua A"), (2, C3, "--", "vòng L2 qua A")]:
        r = np.abs(A).sum() if p == 1 else np.linalg.norm(A)
        X, Y = ball(p)
        ax.plot(X * r, Y * r, color=col, lw=1.9, ls=ls, alpha=.9, label=nm)
    ax.scatter(*q, marker="*", s=330, color=C4, edgecolor="k", zorder=5,
               label="điểm truy vấn q")
    ax.scatter(*A, s=95, color="k", zorder=5)
    ax.text(A[0] + .05, A[1] + .06, "A", fontsize=12, fontweight="bold")
    ax.scatter(*B, s=95, color="k", marker="s", zorder=5)
    ax.text(B[0] + .05, B[1] + .06, "B", fontsize=12, fontweight="bold")
    ax.text(-1.12, -1.38,
            "d(q, A) = 0.85 với CẢ L1 lẫn L2\n"
            "d(q, B): L1 = 1.00  → B NGOÀI hình thoi, xa hơn A\n"
            "         L2 = 0.71  → B TRONG hình tròn, gần hơn A\n"
            "→ đổi metric là đổi luôn hàng xóm gần nhất!",
            fontsize=8.0, color="dimgray",
            bbox=dict(boxstyle="round", fc="#f0fdf4", ec=C3, alpha=.95))
    ax.legend(fontsize=8, loc="upper center", ncol=1)
    ax.set_aspect("equal"); ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.45, 2.25)
    ax.axhline(0, color="k", lw=.7); ax.axvline(0, color="k", lw=.7)
    ax.set_title("Cùng dữ liệu, khác metric,\nkhác hàng xóm gần nhất", fontsize=10)

    fig.suptitle("Hình dạng của khoảng cách quyết định 'ai là hàng xóm'",
                 fontweight="bold")
    fig.tight_layout()
    save(fig, "05_qua_cau_don_vi_minkowski.png")


# ---------------------------------------------------------------- 6
def fig_curse_of_dimensionality():
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    rng = np.random.default_rng(0)

    dims = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000]
    ratio, contrast = [], []
    for d in dims:
        rs = []
        cs = []
        for _ in range(25):
            P = rng.random((500, d))
            q = rng.random(d)
            dist = np.linalg.norm(P - q, axis=1)
            rs.append(dist.min() / dist.max())
            cs.append((dist.max() - dist.min()) / dist.min())
        ratio.append(np.mean(rs)); contrast.append(np.mean(cs))

    dims2 = [2, 5, 10, 20, 50, 100, 200, 500]
    accs = []
    for d in dims2:
        a = []
        for s in range(5):
            r2 = np.random.default_rng(s)
            n = 400
            core = r2.normal(size=(n, 2))
            yv = (core[:, 0] + core[:, 1] > 0).astype(int)
            noise = r2.normal(size=(n, d - 2)) if d > 2 else np.empty((n, 0))
            Xd = np.c_[core, noise]
            Xtr, Xte, ytr, yte = train_test_split(Xd, yv, test_size=.4, random_state=s)
            a.append(KNeighborsClassifier(5).fit(Xtr, ytr).score(Xte, yte))
        accs.append(np.mean(a))

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.1))
    ax = axes[0]
    ax.semilogx(dims, ratio, "o-", color=C1, lw=2)
    ax.axhline(1, color=C2, ls="--", lw=1.5)
    ax.text(1.2, .96, "giới hạn = 1: 'gần nhất' ≈ 'xa nhất'", color=C2, fontsize=8.6)
    ax.set_xlabel("số chiều d"); ax.set_ylabel(r"$d_{min}\,/\,d_{max}$")
    ax.set_ylim(0, 1.08)
    ax.set_title("Tỷ số khoảng cách gần nhất / xa nhất\n→ tiến về 1 khi d tăng", fontsize=9.8)

    ax = axes[1]
    ax.loglog(dims, contrast, "s-", color=C4, lw=2)
    ax.set_xlabel("số chiều d")
    ax.set_ylabel(r"$(d_{max}-d_{min})\,/\,d_{min}$")
    ax.set_title("Độ tương phản khoảng cách sụp đổ\n"
                 "→ khái niệm 'hàng xóm gần' mất ý nghĩa", fontsize=9.8)

    ax = axes[2]
    ax.semilogx(dims2, np.array(accs) * 100, "o-", color=C2, lw=2)
    ax.axhline(50, color="gray", ls=":", lw=1.5)
    ax.text(2.2, 51.5, "đoán bừa (50%)", color="gray", fontsize=8.6)
    ax.set_xlabel("số chiều d (chỉ 2 chiều đầu là hữu ích)")
    ax.set_ylabel("Test accuracy KNN k=5 (%)")
    ax.set_ylim(45, 100)
    ax.set_title("Thêm chiều NHIỄU → accuracy tụt dốc\n"
                 "dù thông tin hữu ích không đổi", fontsize=9.8)

    fig.suptitle("Lời nguyền chiều cao (curse of dimensionality) — kẻ thù số 1 của KNN",
                 fontweight="bold")
    fig.tight_layout()
    save(fig, "06_loi_nguyen_chieu_cao.png")


# ---------------------------------------------------------------- 7
def fig_weights_uniform_vs_distance():
    from sklearn.neighbors import KNeighborsClassifier
    # (a) kịch bản: 2 hàng xóm ĐỎ rất gần, 3 hàng xóm XANH ở xa
    q = np.array([0., 0.])
    near = np.array([[0.34, 0.22], [-0.30, 0.20]])                # lớp ĐỎ
    far = np.array([[1.90, 0.90], [-1.70, 1.30], [0.35, -2.05]])  # lớp XANH
    lab_pos = [(1.45, 0.30), (-1.45, 0.10), (1.90, 1.42),
               (-1.70, 1.82), (0.35, -2.55)]
    P = np.vstack([near, far])
    lab = np.r_[np.ones(2), np.zeros(3)]
    d = np.linalg.norm(P - q, axis=1)
    w = 1 / d

    fig, axes = plt.subplots(1, 4, figsize=(16.4, 4.2))

    ax = axes[0]
    for i, (p_, l_) in enumerate(zip(P, lab)):
        col = C2 if l_ == 1 else C1
        ax.plot([0, p_[0]], [0, p_[1]], color="gray", lw=1.2, ls=":")
        ax.scatter(*p_, s=95, color=col, marker="s" if l_ == 1 else "o", zorder=4)
        near_pt = d[i] < 1
        ax.annotate(f"d = {d[i]:.2f}", xy=p_, xytext=lab_pos[i], fontsize=9.5,
                    color=col, ha="center", va="center",
                    fontweight="bold" if near_pt else "normal",
                    arrowprops=dict(arrowstyle="-", color=col, lw=.8, alpha=.6)
                    if near_pt else None)
    ax.scatter(*q, marker="*", s=360, color=C4, edgecolor="k", zorder=5)
    ax.text(0.02, -0.42, "q", fontsize=12, color=C4, fontweight="bold")
    ax.set_aspect("equal"); ax.set_xlim(-2.9, 2.9); ax.set_ylim(-3.0, 2.25)
    ax.set_title("k = 5 hàng xóm của q:\n2 phiếu ĐỎ rất gần, 3 phiếu XANH ở rất xa",
                 fontsize=9.8)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False); ax.set_frame_on(False)

    ax = axes[1]
    vu = [float((lab == 0).sum()), float((lab == 1).sum())]
    vd = [w[lab == 0].sum(), w[lab == 1].sum()]
    xpos = np.arange(2)
    b1 = ax.bar(xpos - .19, vu, .36, color="#94a3b8", label="uniform: mỗi phiếu = 1")
    b2 = ax.bar(xpos + .19, vd, .36, color=C4, label="distance: phiếu = 1/d")
    for bars in (b1, b2):
        ax.bar_label(bars, fmt="%.2f", fontsize=8.5, padding=2)
    ax.set_xticks(xpos); ax.set_xticklabels(["lớp XANH\n(3 hàng xóm xa)",
                                             "lớp ĐỎ\n(2 hàng xóm gần)"])
    ax.set_ylabel("tổng số phiếu"); ax.set_ylim(0, max(vd) * 1.3)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(f"uniform → XANH thắng (3 > 2)\n"
                 f"distance → ĐỎ thắng ({vd[1]:.2f} > {vd[0]:.2f})", fontsize=9.8)

    # (c)(d) ảnh hưởng lên vùng quyết định
    X, y = _moons(90, 0.36, 3)
    for ax, wname, ttl in zip(axes[2:], ["uniform", "distance"],
                              ["weights='uniform', k=25\nmọi hàng xóm nói to như nhau",
                               "weights='distance', k=25\nhàng xóm gần có tiếng nói lớn hơn"]):
        m = KNeighborsClassifier(n_neighbors=25, weights=wname).fit(X, y)
        _boundary(ax, m, X, y)
        for c in ax.collections[-2:]:
            c.set_sizes([34])
        ax.set_title(ttl, fontsize=9.8)
    axes[3].set_xlabel("→ ranh giới bám sát điểm train hơn, gồ ghề hơn\n"
                       "(và train accuracy luôn = 100%)", fontsize=8.4, color="dimgray")

    fig.suptitle("weights='uniform' vs weights='distance' — "
                 "khi hàng xóm ở xa, trọng số khoảng cách đổi cả kết quả",
                 fontweight="bold")
    fig.tight_layout()
    save(fig, "07_uniform_vs_distance.png")


# ---------------------------------------------------------------- 8
def fig_voronoi():
    from scipy.spatial import Voronoi, voronoi_plot_2d
    rng = np.random.default_rng(4)
    P = np.vstack([rng.normal([3, 3], .95, (11, 2)),
                   rng.normal([6.4, 5.6], .95, (11, 2))])
    lab = np.r_[np.zeros(11), np.ones(11)]

    xx, yy = np.meshgrid(np.linspace(0, 9.5, 460), np.linspace(0, 9.0, 440))
    G = np.c_[xx.ravel(), yy.ravel()]
    nearest = np.argmin(((G[:, None, :] - P[None]) ** 2).sum(-1), axis=1)
    Z = lab[nearest].reshape(xx.shape)

    vor = Voronoi(P)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0))
    for ax, show_cells in zip(axes, [True, False]):
        ax.pcolormesh(xx, yy, Z, cmap=CMAP_BG, shading="auto", zorder=0)
        if show_cells:
            voronoi_plot_2d(vor, ax=ax, show_points=False, show_vertices=False,
                            line_colors="gray", line_width=1.0, line_alpha=.9)
        ax.contour(xx, yy, Z, levels=[.5], colors="k", linewidths=2.2, zorder=2)
        ax.scatter(P[lab == 0, 0], P[lab == 0, 1], s=52, color=C1, edgecolor="w",
                   linewidth=.6, zorder=4)
        ax.scatter(P[lab == 1, 0], P[lab == 1, 1], s=52, color=C2, marker="s",
                   edgecolor="w", linewidth=.6, zorder=4)
        ax.set_xlim(0, 9.5); ax.set_ylim(0, 9.0)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    axes[0].set_title("Sơ đồ Voronoi: mỗi điểm train sở hữu một ô\n"
                      "(tập các vị trí mà NÓ là điểm gần nhất)", fontsize=10)
    axes[1].set_title("1-NN decision boundary = gộp các ô cùng nhãn\n"
                      "→ ranh giới là đường gấp khúc đa giác", fontsize=10)
    fig.suptitle("1-NN chính là sơ đồ Voronoi — thêm/bớt MỘT điểm train là đổi ngay ranh giới "
                 "(vì sao k=1 có variance rất cao)", fontweight="bold", fontsize=11)
    fig.tight_layout()
    save(fig, "08_voronoi_1nn.png")


# ---------------------------------------------------------------- 9
def fig_knn_regression():
    from sklearn.neighbors import KNeighborsRegressor
    rng = np.random.default_rng(2)
    x = np.sort(rng.uniform(0, 2 * np.pi, 60))
    y = np.sin(x) + rng.normal(0, .22, 60)
    xs = np.linspace(0, 2 * np.pi, 600).reshape(-1, 1)

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.7))
    cfg = [(1, "uniform", "k = 1 — nối thẳng qua từng điểm\nOVERFIT hoàn toàn", C2),
           (5, "uniform", "k = 5 — bậc thang vừa phải\nvừa đẹp", C3),
           (25, "uniform", "k = 25 — bậc thang thô,\nbẹp ở hai đầu (bias cao)", C1),
           (5, "distance", "k = 5, weights='distance'\nliên tục, đi ĐÚNG qua từng điểm train", C4)]
    for ax, (k, wname, ttl, col) in zip(axes, cfg):
        m = KNeighborsRegressor(n_neighbors=k, weights=wname).fit(x.reshape(-1, 1), y)
        ax.scatter(x, y, s=22, color="k", alpha=.55, label="dữ liệu train")
        ax.plot(xs, np.sin(xs), "--", color="gray", lw=1.5, label="hàm thật sin(x)")
        ax.plot(xs, m.predict(xs), color=col, lw=2.2, label="KNN dự đoán")
        ax.set_ylim(-1.9, 1.9)
        ax.set_title(ttl, fontsize=9.5)
        ax.set_xlabel("x")
    axes[0].set_ylabel("y"); axes[0].legend(fontsize=7.5, loc="lower left")
    fig.suptitle("KNeighborsRegressor: dự đoán = TRUNG BÌNH y của k hàng xóm "
                 "→ đường dự đoán luôn là hàm bậc thang", fontweight="bold")
    fig.tight_layout()
    save(fig, "09_knn_hoi_quy_bac_thang.png")


if __name__ == "__main__":
    fig_how_knn_works()
    fig_boundary_by_k()
    fig_bias_variance_k()
    fig_why_scale()
    fig_minkowski_balls()
    fig_curse_of_dimensionality()
    fig_weights_uniform_vs_distance()
    fig_voronoi()
    fig_knn_regression()
    print("Xong.")
