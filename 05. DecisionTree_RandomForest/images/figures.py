"""
Sinh toàn bộ hình minh hoạ cho Lab 05 - Decision Tree & Random Forest.

Chạy:  python figures.py
Kết quả: các file .png trong cùng thư mục, được nhúng vào notebook bằng markdown.
Sinh viên có thể sửa script này để tự thí nghiệm với tham số (đổi max_depth,
đổi ccp_alpha, đổi số cây...) rồi chạy lại để xem hình thay đổi thế nào.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 110, "savefig.bbox": "tight",
    "font.size": 10, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})
C1, C2, C3, C4 = "#2563eb", "#dc2626", "#059669", "#d97706"
CMAP_BG = ListedColormap(["#dbeafe", "#fee2e2"])
CMAP_PROBA = LinearSegmentedColormap.from_list(
    "proba", ["#93c5fd", "#dbeafe", "#ffffff", "#fee2e2", "#fca5a5"])


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p)
    plt.close(fig)
    print("wrote", name)


def _entropy(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def _boundary(ax, model, X, y, h=0.02, proba=False):
    x0, x1 = X[:, 0].min() - .5, X[:, 0].max() + .5
    y0, y1 = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x0, x1, h), np.arange(y0, y1, h))
    G = np.c_[xx.ravel(), yy.ravel()]
    if proba:
        Z = model.predict_proba(G)[:, 1].reshape(xx.shape)
        ax.contourf(xx, yy, Z, levels=np.linspace(0, 1, 21), cmap=CMAP_PROBA,
                    zorder=0)
        ax.contour(xx, yy, Z, levels=[.5], colors="k", linewidths=1.0, zorder=1)
    else:
        Z = model.predict(G).reshape(xx.shape)
        ax.pcolormesh(xx, yy, Z, cmap=CMAP_BG, shading="auto", zorder=0)
        ax.contour(xx, yy, Z, levels=[.5], colors="k", linewidths=1.2, zorder=1)
    ms = 26 if proba else 16
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=ms, color=C1, edgecolor="w",
               linewidth=.5, zorder=3)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=ms, color=C2, marker="s",
               edgecolor="w", linewidth=.5, zorder=3)
    ax.set_xlim(x0, x1 - h); ax.set_ylim(y0, y1 - h)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)


# ---------------------------------------------------------------- 1
def fig_impurity_criteria():
    p = np.linspace(0, 1, 500)
    H = _entropy(p)
    G = 2 * p * (1 - p)
    E = np.minimum(p, 1 - p)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.5))
    ax = axes[0]
    ax.plot(p, H, color=C1, lw=2.4, label="Entropy  $-p\\log_2 p-(1-p)\\log_2(1-p)$")
    ax.plot(p, G, color=C3, lw=2.4, label="Gini  $2p(1-p)$")
    ax.plot(p, E, color=C2, lw=2.4, label="Misclassification  $\\min(p, 1-p)$")
    ax.axvline(.5, color="gray", ls=":", lw=1.4)
    ax.scatter([.5, .5, .5], [1, .5, .5], s=45, color="k", zorder=5)
    ax.annotate("cực đại tại p = 0.5\n(nút hỗn loạn nhất)", xy=(.5, 1.0),
                xytext=(.54, 1.22), fontsize=9, color="dimgray",
                arrowprops=dict(arrowstyle="->", color="gray"))
    ax.annotate("= 0 tại p = 0 và p = 1  (nút THUẦN → dừng chia)", xy=(0, 0),
                xytext=(.13, .07), fontsize=8.8, color="dimgray",
                arrowprops=dict(arrowstyle="->", color="gray"))
    ax.set_xlabel("p = tỷ lệ lớp dương trong nút")
    ax.set_ylabel("độ vẩn đục (impurity)")
    ax.set_ylim(-.03, 1.62); ax.legend(fontsize=8.4, loc="upper center")
    ax.set_title("Ba tiêu chí impurity cho bài toán 2 lớp", fontsize=10.5)

    ax = axes[1]
    ax.plot(p, H / 2, color=C1, lw=2.6, label="Entropy / 2 (chuẩn hoá về cùng đỉnh)")
    ax.plot(p, G, color=C3, lw=2.6, ls="--", label="Gini")
    ax.plot(p, E, color=C2, lw=2.0, ls=":", label="Misclassification")
    ax.fill_between(p, H / 2, G, color=C4, alpha=.25)
    ax.set_xlabel("p"); ax.set_ylabel("impurity (đã chuẩn hoá)")
    ax.set_ylim(-.03, .98); ax.legend(fontsize=8.4, loc="lower center")
    ax.set_title("Entropy và Gini gần như TRÙNG NHAU\n"
                 "→ đổi `criterion` hiếm khi đổi kết quả", fontsize=10.5)
    ax.text(.045, .95,
            "Entropy & Gini LÕM CHẶT (strictly concave)\n"
            "→ mọi phép chia không tầm thường đều cho gain > 0.\n"
            "Misclassification chỉ tuyến tính từng khúc\n"
            "→ gain có thể = 0 dù phép chia rất hữu ích\n"
            "→ KHÔNG dùng để mọc cây, chỉ dùng để cắt tỉa.",
            fontsize=8.2, color="#334155", va="top",
            bbox=dict(boxstyle="round", fc="#f8fafc", ec="#94a3b8", alpha=.95))
    fig.suptitle("Impurity đo 'độ hỗn loạn' của một nút — cây luôn chia sao cho impurity giảm nhiều nhất",
                 fontweight="bold")
    fig.tight_layout()
    save(fig, "01_entropy_gini_misclassification.png")


# ---------------------------------------------------------------- 2
def fig_one_split():
    rng = np.random.default_rng(7)
    x0 = rng.normal(3.2, 1.15, 24)
    x1 = rng.normal(7.6, 1.30, 26)
    x = np.r_[x0, x1]
    y = np.r_[np.zeros(24), np.ones(26)]
    o = np.argsort(x); x, y = x[o], y[o]
    N = len(x)

    thr = (x[:-1] + x[1:]) / 2
    ig, gg = [], []
    p_par = y.mean()
    H_par, G_par = _entropy(p_par), 2 * p_par * (1 - p_par)
    for t in thr:
        L, R = y[x < t], y[x >= t]
        wl, wr = len(L) / N, len(R) / N
        ig.append(H_par - wl * _entropy(L.mean()) - wr * _entropy(R.mean()))
        gg.append(G_par - wl * 2 * L.mean() * (1 - L.mean())
                  - wr * 2 * R.mean() * (1 - R.mean()))
    ig, gg = np.array(ig), np.array(gg)
    best = thr[int(np.argmax(ig))]
    L, R = y[x < best], y[x >= best]

    fig = plt.figure(figsize=(15.5, 4.4))
    ax = fig.add_subplot(1, 3, 1)
    ax.scatter(x[y == 0], np.zeros((y == 0).sum()) + .06 * rng.normal(0, 1, (y == 0).sum()),
               s=48, color=C1, alpha=.85, label="lớp 0")
    ax.scatter(x[y == 1], np.zeros((y == 1).sum()) + .06 * rng.normal(0, 1, (y == 1).sum()),
               s=48, color=C2, marker="s", alpha=.85, label="lớp 1")
    for t in thr[::4]:
        ax.axvline(t, color="gray", lw=.6, alpha=.5)
    ax.axvline(best, color=C3, lw=2.6)
    ax.text(best + .15, .30, f"ngưỡng tốt nhất\nx = {best:.2f}", color=C3, fontsize=9.5)
    ax.set_yticks([]); ax.set_ylim(-.45, .45)
    ax.set_xlabel("giá trị feature x")
    ax.legend(fontsize=8.5, loc="lower left")
    ax.set_title("Bước 1: liệt kê MỌI ngưỡng ứng viên\n(trung điểm giữa 2 giá trị liền kề)",
                 fontsize=10)

    ax = fig.add_subplot(1, 3, 2)
    ax.plot(thr, ig, color=C1, lw=2.2, label="Information Gain (entropy)")
    ax.plot(thr, gg, color=C3, lw=2.2, ls="--", label="Gini gain")
    ax.axvline(best, color=C3, lw=1.8, ls=":")
    ax.scatter([best], [ig.max()], s=90, color=C2, zorder=5)
    ax.annotate(f"max IG = {ig.max():.3f}\ntại x = {best:.2f}", xy=(best, ig.max()),
                xytext=(best - 4.4, ig.max() * .72), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="gray"))
    ax.set_xlabel("ngưỡng t"); ax.set_ylabel("mức giảm impurity")
    ax.legend(fontsize=8.5, loc="upper right")
    ax.set_title("Bước 2: chấm điểm từng ngưỡng\n→ chọn ngưỡng cho gain LỚN NHẤT", fontsize=10)

    ax = fig.add_subplot(1, 3, 3)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.axis("off")

    def box(cx, cy, w, h, txt, fc, ec):
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                    boxstyle="round,pad=0.18", fc=fc, ec=ec, lw=1.6))
        ax.text(cx, cy, txt, ha="center", va="center", fontsize=8.8)

    n0, n1 = int((y == 0).sum()), int((y == 1).sum())
    box(5, 8.4, 5.2, 2.0,
        f"NÚT CHA:  N = {N}\n[{n0} lớp 0, {n1} lớp 1]\n"
        f"entropy = {H_par:.3f}   gini = {G_par:.3f}", "#f1f5f9", "#475569")
    ax.text(5, 6.60, f"x < {best:.2f} ?", ha="center", va="center", fontsize=10,
            color=C3, fontweight="bold")
    ax.plot([5, 2.3], [7.4, 5.3], color="#475569", lw=1.4)
    ax.plot([5, 7.7], [7.4, 5.3], color="#475569", lw=1.4)
    ax.text(2.55, 5.92, "Đúng", fontsize=9, color="#475569")
    ax.text(7.20, 5.92, "Sai", fontsize=9, color="#475569")
    box(2.3, 4.0, 4.3, 2.2,
        f"CON TRÁI: N = {len(L)}\n[{int((L == 0).sum())} lớp 0, {int((L == 1).sum())} lớp 1]\n"
        f"entropy = {_entropy(L.mean()):.3f}", "#dbeafe", C1)
    box(7.7, 4.0, 4.3, 2.2,
        f"CON PHẢI: N = {len(R)}\n[{int((R == 0).sum())} lớp 0, {int((R == 1).sum())} lớp 1]\n"
        f"entropy = {_entropy(R.mean()):.3f}", "#fee2e2", C2)
    ax.text(5, 1.35,
            f"IG = {H_par:.3f} − ({len(L)}/{N})·{_entropy(L.mean()):.3f} "
            f"− ({len(R)}/{N})·{_entropy(R.mean()):.3f} = {ig.max():.3f}",
            ha="center", fontsize=9.6, color="#334155",
            bbox=dict(boxstyle="round", fc="#f0fdf4", ec=C3))
    ax.set_title("Bước 3: cây con thu được sau MỘT bước chia", fontsize=10)

    fig.suptitle("CART làm gì ở MỘT nút: thử mọi (feature, ngưỡng) → chọn cái làm impurity giảm nhiều nhất",
                 fontweight="bold")
    fig.tight_layout()
    save(fig, "02_mot_buoc_split.png")


# ---------------------------------------------------------------- 3
def fig_axis_aligned():
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(3)
    n = 260
    X = rng.uniform(-3, 3, (n, 2))

    # dữ liệu KHÔNG nhiễu, để số lá phản ánh đúng hình học của ranh giới
    y_axis = (X[:, 0] > 0).astype(int)               # ranh giới song song trục
    y_diag = (X[:, 1] > X[:, 0]).astype(int)         # ranh giới CHÉO

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.5))
    dt1 = DecisionTreeClassifier(random_state=0).fit(X, y_axis)
    _boundary(axes[0], dt1, X, y_axis)
    axes[0].axvline(0, color=C4, lw=2.4, ls="--", zorder=4)
    axes[0].set_title(f"Ranh giới THẲNG ĐỨNG — cây khớp HOÀN HẢO\n"
                      f"chỉ tốn {dt1.get_n_leaves()} lá, sâu {dt1.get_depth()}",
                      fontsize=9.8)

    dt2 = DecisionTreeClassifier(random_state=0).fit(X, y_diag)
    _boundary(axes[1], dt2, X, y_diag)
    axes[1].plot([-3.5, 3.5], [-3.5, 3.5], color=C4, lw=2.4, ls="--", zorder=4)
    axes[1].set_title(f"Ranh giới CHÉO — cây phải leo bậc thang\n"
                      f"({dt2.get_n_leaves()} lá, sâu {dt2.get_depth()}) để xấp xỉ 1 đường thẳng",
                      fontsize=9.8)

    lr = LogisticRegression().fit(X, y_diag)
    _boundary(axes[2], lr, X, y_diag)
    axes[2].plot([-3.5, 3.5], [-3.5, 3.5], color=C4, lw=2.4, ls="--", zorder=4)
    axes[2].set_title("Cùng dữ liệu, Logistic Regression\nchỉ cần ĐÚNG 1 đường thẳng", fontsize=9.8)

    fig.suptitle("Điểm yếu cố hữu của Decision Tree: mọi phép chia đều SONG SONG TRỤC "
                 "(axis-aligned)", fontweight="bold")
    fig.tight_layout()
    save(fig, "03_ranh_gioi_bac_thang.png")


# ---------------------------------------------------------------- 4
def fig_max_depth():
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split
    X, y = make_moons(n_samples=500, noise=0.30, random_state=1)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.4, random_state=0, stratify=y)

    fig = plt.figure(figsize=(16.5, 3.9))
    cfg = [(1, "max_depth = 1 (decision stump)\n1 nhát cắt — UNDERFIT nặng"),
           (3, "max_depth = 3\nbắt được hình dạng chính"),
           (6, "max_depth = 6\nbắt đầu mọc thêm ô nhỏ"),
           (None, "max_depth = None\nmỗi ô nhiễu một hộp — OVERFIT")]
    for i, (d, ttl) in enumerate(cfg):
        ax = fig.add_subplot(1, 5, i + 1)
        m = DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xtr, ytr)
        _boundary(ax, m, Xtr, ytr)
        ax.set_title(f"{ttl}\ntrain {m.score(Xtr, ytr)*100:.0f}% / test "
                     f"{m.score(Xte, yte)*100:.0f}% / {m.get_n_leaves()} lá", fontsize=8.8)

    depths = list(range(1, 21))
    tr = [DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xtr, ytr).score(Xtr, ytr)
          for d in depths]
    te = [DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xtr, ytr).score(Xte, yte)
          for d in depths]
    ax = fig.add_subplot(1, 5, 5)
    ax.plot(depths, np.array(tr) * 100, "o-", color=C1, ms=3.5, label="Train")
    ax.plot(depths, np.array(te) * 100, "s-", color=C2, ms=3.5, label="Test")
    b = depths[int(np.argmax(te))]
    ax.axvline(b, color=C3, ls="--", lw=1.6)
    ax.text(b + .5, 74, f"depth tốt nhất = {b}", color=C3, fontsize=8.4)
    ax.fill_between(depths, np.array(te) * 100, np.array(tr) * 100,
                    color="#94a3b8", alpha=.20)
    ax.set_xlabel("max_depth"); ax.set_ylabel("Accuracy (%)")
    ax.legend(fontsize=8); ax.set_ylim(70, 102)
    ax.set_title("Train leo mãi lên 100%,\ntest quay đầu → OVERFIT", fontsize=9)
    fig.suptitle("`max_depth` là nút vặn quan trọng nhất của Decision Tree",
                 fontweight="bold")
    fig.tight_layout()
    save(fig, "04_anh_huong_max_depth.png")


# ---------------------------------------------------------------- 5
def fig_ccp_pruning():
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3, random_state=0, stratify=y)

    base = DecisionTreeClassifier(random_state=0)
    path = base.cost_complexity_pruning_path(Xtr, ytr)
    alphas = path.ccp_alphas[:-1]           # bỏ alpha cuối (cây chỉ còn 1 nút)
    models = [DecisionTreeClassifier(random_state=0, ccp_alpha=a).fit(Xtr, ytr)
              for a in alphas]
    tr = [m.score(Xtr, ytr) for m in models]
    te = [m.score(Xte, yte) for m in models]
    leaves = [m.get_n_leaves() for m in models]
    depth = [m.get_depth() for m in models]
    ibest = int(np.argmax(te))

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.5))
    ax = axes[0]
    ax.plot(alphas, np.array(tr) * 100, "o-", color=C1, ms=3.5, label="Train accuracy")
    ax.plot(alphas, np.array(te) * 100, "s-", color=C2, ms=3.5, label="Test accuracy")
    ax.axvline(alphas[ibest], color=C3, ls="--", lw=1.8)
    ax.annotate(f"ccp_alpha tốt nhất ≈ {alphas[ibest]:.4f}\n"
                f"test = {te[ibest]*100:.1f}%  ({leaves[ibest]} lá thay vì {leaves[0]})",
                xy=(alphas[ibest], te[ibest] * 100), xytext=(alphas[ibest] * 2.1, 95.2),
                fontsize=9, color="#334155",
                arrowprops=dict(arrowstyle="->", color="gray"))
    ax.set_xlabel(r"$\alpha$ (ccp_alpha) — phạt càng nặng, cây càng nhỏ  →")
    ax.set_ylabel("Accuracy (%)"); ax.legend(fontsize=9, loc="lower left")
    ax.set_title("Cắt tỉa vừa đủ làm TEST accuracy TĂNG\n"
                 "(cây đầy đủ train 100% nhưng test kém hơn)", fontsize=10)

    ax = axes[1]
    ax.plot(alphas, leaves, "o-", color=C4, ms=3.5, label="số lá $|T|$")
    ax.set_xlabel(r"$\alpha$ (ccp_alpha)"); ax.set_ylabel("số lá", color=C4)
    ax.tick_params(axis="y", labelcolor=C4)
    ax2 = ax.twinx(); ax2.grid(False)
    ax2.plot(alphas, depth, "s--", color="#7c3aed", ms=3.5, label="độ sâu")
    ax2.set_ylabel("độ sâu cây", color="#7c3aed")
    ax2.tick_params(axis="y", labelcolor="#7c3aed")
    ax.axvline(alphas[ibest], color=C3, ls="--", lw=1.8)
    ax.set_title(r"Tăng $\alpha$ → cây co lại đơn điệu"
                 "\n(mỗi bậc thang là một nhánh bị cắt)", fontsize=10)

    fig.suptitle(r"Cost-complexity pruning:  $R_\alpha(T) = R(T) + \alpha\,|T|$"
                 "  — trả tiền cho mỗi chiếc lá", fontweight="bold")
    fig.tight_layout()
    save(fig, "05_cost_complexity_pruning.png")


# ---------------------------------------------------------------- 6
def fig_bagging_oob():
    rng = np.random.default_rng(1)
    N, B = 10, 4
    draws = [rng.integers(0, N, N) for _ in range(B)]
    counts = np.array([[int((d == i).sum()) for i in range(N)] for d in draws])

    fig = plt.figure(figsize=(14.2, 4.8))
    ax = fig.add_subplot(1, 2, 1)
    ax.set_xlim(-1.5, B + 1.1); ax.set_ylim(-1.4, N + .9)
    ax.axis("off")
    for i in range(N):
        yv = N - 1 - i
        ax.add_patch(Rectangle((-1.35, yv + .1), .95, .8, fc="#e2e8f0", ec="#475569"))
        ax.text(-.875, yv + .5, f"#{i+1}", ha="center", va="center", fontsize=8.5)
    ax.text(-.875, N + .45, "Tập gốc", ha="center", fontsize=9.5, fontweight="bold")
    for b in range(B):
        ax.text(b + .5, N + .45, f"Cây {b+1}", ha="center", fontsize=9.5, fontweight="bold")
        for i in range(N):
            yv = N - 1 - i
            c = counts[b, i]
            if c == 0:
                fc, ec, txt, tc = "#ffffff", "#dc2626", "OOB", C2
            else:
                fc, ec, txt, tc = ["#dbeafe", "#93c5fd", "#3b82f6"][min(c - 1, 2)], \
                                  "#1d4ed8", f"×{c}", "#111827"
            ax.add_patch(Rectangle((b + .05, yv + .1), .9, .8, fc=fc, ec=ec,
                                   lw=1.5 if c == 0 else 1.0,
                                   ls="--" if c == 0 else "-"))
            ax.text(b + .5, yv + .5, txt, ha="center", va="center", fontsize=7.6, color=tc)
        n_oob = int((counts[b] == 0).sum())
        ax.text(b + .5, -.75, f"{n_oob}/{N} OOB", ha="center", fontsize=8.6, color=C2)
    ax.set_title("Bootstrap: mỗi cây bốc N mẫu CÓ HOÀN LẠI\n"
                 "→ vài mẫu bị lấy 2–3 lần, vài mẫu KHÔNG được lấy lần nào (OOB)",
                 fontsize=10)

    ax = fig.add_subplot(1, 2, 2)
    Ns = np.arange(2, 201)
    theo = (1 - 1 / Ns) ** Ns
    emp = []
    for n in Ns[::8]:
        s = [((np.bincount(rng.integers(0, n, n), minlength=n) == 0).mean())
             for _ in range(120)]
        emp.append(np.mean(s))
    ax.plot(Ns, theo * 100, color=C1, lw=2.4, label=r"lý thuyết $(1-1/N)^N$")
    ax.plot(Ns[::8], np.array(emp) * 100, "o", color=C4, ms=4.5, alpha=.85,
            label="mô phỏng thực tế")
    ax.axhline(100 / np.e, color=C2, ls="--", lw=1.8)
    ax.text(78, 100 / np.e + 1.1, r"$1/e \approx 36.8\%$", color=C2, fontsize=11)
    ax.set_xlabel("N (số mẫu trong tập train)")
    ax.set_ylabel("tỷ lệ mẫu KHÔNG được cây này nhìn thấy (%)")
    ax.set_ylim(22, 44); ax.legend(fontsize=9, loc="lower right")
    ax.set_title("Xác suất một mẫu là OOB hội tụ rất nhanh về 1/e\n"
                 "→ mỗi cây bỏ sót ~36.8% dữ liệu = tập validation MIỄN PHÍ", fontsize=10)

    fig.suptitle("Bagging và Out-of-Bag: nguồn ngẫu nhiên thứ nhất của Random Forest",
                 fontweight="bold")
    fig.tight_layout()
    save(fig, "06_bagging_va_oob.png")


# ---------------------------------------------------------------- 7
def fig_rf_smooths():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=300, noise=0.26, random_state=2)

    fig, axes = plt.subplots(1, 4, figsize=(15.8, 4.0))
    for ax, n in zip(axes, [1, 5, 25, 200]):
        m = RandomForestClassifier(n_estimators=n, random_state=0).fit(X, y)
        _boundary(ax, m, X, y, proba=True)
        ax.set_title(f"{n} cây" + ("  (= 1 Decision Tree)" if n == 1 else ""),
                     fontsize=10.5)
    axes[0].set_xlabel("ranh giới sắc lẹm, đầy góc cạnh", fontsize=8.6, color="dimgray")
    axes[3].set_xlabel("ranh giới mượt, có vùng 'lưỡng lự' (màu nhạt)",
                       fontsize=8.6, color="dimgray")
    fig.suptitle("Càng nhiều cây, biên quyết định càng mượt — màu càng nhạt nghĩa là "
                 "xác suất càng gần 0.5 (rừng đang phân vân)", fontweight="bold")
    fig.tight_layout()
    save(fig, "07_random_forest_lam_muot.png")


# ---------------------------------------------------------------- 8
def fig_n_estimators():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    X, y = make_classification(n_samples=2500, n_features=25, n_informative=8,
                               n_redundant=5, flip_y=.10, class_sep=.9,
                               random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.35, random_state=0, stratify=y)

    ns = [1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 300, 500]
    tr, te, oob = [], [], []
    for n in ns:
        m = RandomForestClassifier(n_estimators=n, oob_score=(n > 10),
                                   random_state=0, n_jobs=-1).fit(Xtr, ytr)
        tr.append(m.score(Xtr, ytr)); te.append(m.score(Xte, yte))
        oob.append(m.oob_score_ if n > 10 else np.nan)
    dt = DecisionTreeClassifier(random_state=0).fit(Xtr, ytr)

    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    ax.semilogx(ns, np.array(tr) * 100, "o-", color=C1, ms=4, label="Train accuracy (RF)")
    ax.semilogx(ns, np.array(te) * 100, "s-", color=C2, ms=4, label="Test accuracy (RF)")
    ax.semilogx(ns, np.array(oob) * 100, "^--", color=C3, ms=4,
                label="OOB score (validation miễn phí)")
    ax.axhline(dt.score(Xte, yte) * 100, color="#7c3aed", ls=":", lw=2,
               label=f"1 Decision Tree không giới hạn: {dt.score(Xte, yte)*100:.1f}%")
    ax.annotate("tăng rất nhanh\nở 20 cây đầu", xy=(8, te[4] * 100), xytext=(1.35, 90),
                fontsize=9, color="dimgray", arrowprops=dict(arrowstyle="->", color="gray"))
    ax.annotate("bão hoà: thêm cây KHÔNG làm test tệ đi\n"
                "(RF không overfit theo số cây)", xy=(320, te[-2] * 100),
                xytext=(18, 95.4), fontsize=9, color="dimgray",
                arrowprops=dict(arrowstyle="->", color="gray"))
    ax.set_xlabel("n_estimators (số cây) — thang log")
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(66, 103)
    ax.legend(fontsize=8.6, loc="lower center", ncol=2)
    ax.set_title("Thêm cây chỉ có LỢI (rồi bão hoà) — khác hẳn tăng `max_depth`\n"
                 "OOB score bám sát test accuracy mà không tốn một mẫu nào",
                 fontweight="bold", fontsize=11)
    save(fig, "08_test_acc_theo_so_cay.png")


# ---------------------------------------------------------------- 9
def fig_importance_bias():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(0)
    n = 1500
    x_bin = rng.integers(0, 2, n)                       # THẬT SỰ hữu ích, chỉ 2 giá trị
    x_ord = rng.integers(0, 4, n)                       # hữu ích vừa, 4 giá trị
    x_noise_cont = rng.normal(size=n)                   # NHIỄU, ~1500 giá trị khác nhau
    x_noise_id = rng.permutation(n).astype(float)       # NHIỄU, ID duy nhất từng dòng
    x_noise_bin = rng.integers(0, 2, n).astype(float)   # NHIỄU, chỉ 2 giá trị

    logit = 2.6 * (x_bin - .5) + 0.55 * (x_ord - 1.5)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    X = np.c_[x_bin, x_ord, x_noise_cont, x_noise_id, x_noise_bin]
    names = ["x_bin\n(HỮU ÍCH, 2 giá trị)", "x_ord\n(hữu ích, 4 giá trị)",
             "nhiễu_liên_tục\n(1500 giá trị)", "nhiễu_ID\n(1500 giá trị)",
             "nhiễu_nhị_phân\n(2 giá trị)"]
    good = [True, True, False, False, False]

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.35, random_state=0, stratify=y)
    rf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1).fit(Xtr, ytr)
    imp = rf.feature_importances_
    perm = permutation_importance(rf, Xte, yte, n_repeats=25, random_state=0, n_jobs=-1)

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.6))
    ypos = np.arange(len(names))
    cols = [C3 if g else C2 for g in good]
    axes[0].barh(ypos, imp, color=cols, alpha=.85)
    axes[0].set_yticks(ypos); axes[0].set_yticklabels(names, fontsize=8.4)
    axes[0].invert_yaxis(); axes[0].set_xlabel("feature_importances_ (impurity-based)")
    axes[0].set_title("BỊ THIÊN VỊ: feature nhiễu có NHIỀU giá trị\n"
                      "vẫn được chấm điểm cao", fontsize=10)
    for i, v in enumerate(imp):
        axes[0].text(v + .004, i, f"{v:.3f}", va="center", fontsize=8.4)

    axes[1].barh(ypos, perm.importances_mean, xerr=perm.importances_std,
                 color=cols, alpha=.85, error_kw=dict(lw=1, ecolor="#475569"))
    axes[1].set_yticks(ypos); axes[1].set_yticklabels(names, fontsize=8.4)
    axes[1].invert_yaxis()
    axes[1].axvline(0, color="k", lw=1)
    axes[1].set_xlabel("permutation_importance (đo trên tập TEST)")
    axes[1].set_title("TRUNG THỰC: feature nhiễu tụt về ~0\n"
                      "vì xáo trộn chúng không làm accuracy giảm", fontsize=10)
    for i, v in enumerate(perm.importances_mean):
        axes[1].text(v + .004, i, f"{v:.3f}", va="center", fontsize=8.4)

    fig.suptitle("Xanh = feature THẬT SỰ hữu ích, Đỏ = feature NHIỄU thuần tuý  —  "
                 "hai cách đo cho hai câu trả lời rất khác nhau", fontweight="bold")
    fig.tight_layout()
    save(fig, "09_feature_importance_thien_vi.png")


if __name__ == "__main__":
    fig_impurity_criteria()
    fig_one_split()
    fig_axis_aligned()
    fig_max_depth()
    fig_ccp_pruning()
    fig_bagging_oob()
    fig_rf_smooths()
    fig_n_estimators()
    fig_importance_bias()
    print("Xong.")
