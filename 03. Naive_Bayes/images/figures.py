"""
Sinh toàn bộ hình minh hoạ cho Lab 03 - Naive Bayes.

Chạy:  python figures.py
Kết quả: các file .png trong cùng thư mục, được nhúng vào notebook bằng markdown.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 110, "savefig.bbox": "tight",
    "font.size": 10, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})
C1, C2, C3, C4 = "#2563eb", "#dc2626", "#059669", "#d97706"


def save(fig, name):
    fig.savefig(os.path.join(OUT, name)); plt.close(fig); print("wrote", name)


# ---------------------------------------------------------------- 1
def fig_bayes_theorem():
    """Nghịch lý xét nghiệm: minh hoạ bằng tần suất tự nhiên trên 10.000 người."""
    N, prev, sens, spec = 10000, 0.01, 0.99, 0.95
    sick = int(N * prev)                       # 100
    tp = int(sick * sens)                      # 99
    fn = sick - tp                             # 1
    healthy = N - sick                         # 9900
    fp = int(healthy * (1 - spec))             # 495
    tn = healthy - fp

    fig = plt.figure(figsize=(14, 4.6))

    # (a) lưới 100x100 người
    ax = fig.add_subplot(1, 3, 1)
    grid = np.zeros((100, 100))
    flat = grid.ravel()
    flat[:tp] = 3; flat[tp:tp + fn] = 2; flat[tp + fn:tp + fn + fp] = 1
    rng = np.random.default_rng(0)
    idx = np.arange(N); rng.shuffle(idx)
    img = np.zeros(N); img[idx] = flat
    ax.imshow(img.reshape(100, 100), cmap=matplotlib.colors.ListedColormap(
        ["#e5e7eb", "#fca5a5", "#fde68a", "#dc2626"]), interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_title("10.000 người đi xét nghiệm", fontsize=10)
    handles = [Rectangle((0, 0), 1, 1, color=c) for c in ["#e5e7eb", "#fca5a5", "#fde68a", "#dc2626"]]
    ax.legend(handles, [f"khoẻ, âm tính ({tn})", f"khoẻ, DƯƠNG TÍNH GIẢ ({fp})",
                        f"bệnh, âm tính giả ({fn})", f"bệnh, dương tính đúng ({tp})"],
              fontsize=7.2, loc="upper center", bbox_to_anchor=(.5, -.02), ncol=2)

    # (b) cột: ai thực sự dương tính
    ax = fig.add_subplot(1, 3, 2)
    ax.bar(["Dương tính giả\n(khoẻ)", "Dương tính đúng\n(bệnh)"], [fp, tp],
           color=["#fca5a5", "#dc2626"])
    ax.set_ylabel("số người")
    ax.set_title(f"Trong {fp + tp} người có kết quả DƯƠNG TÍNH…", fontsize=10)
    for i, v in enumerate([fp, tp]):
        ax.text(i, v + 12, str(v), ha="center", fontweight="bold")
    ax.text(.5, fp * .62, f"chỉ {tp/(tp+fp)*100:.1f}% thật sự có bệnh!",
            ha="center", fontsize=11, color=C2, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=.94,
                      edgecolor=C2, linewidth=1.2))
    ax.set_ylim(0, fp * 1.25)

    # (c) công thức
    ax = fig.add_subplot(1, 3, 3); ax.axis("off")
    ax.text(0, .97, "Định lý Bayes làm rõ nghịch lý", fontsize=11.5, fontweight="bold", va="top")
    ax.text(0, .80,
            "$P(\\mathrm{bệnh}\\,|\\,+)=\\dfrac{P(+\\,|\\,\\mathrm{bệnh})\;P(\\mathrm{bệnh})}"
            "{P(+)}$", fontsize=15, va="top")
    ax.text(0, .58,
            "$=\\dfrac{0.99 \\times 0.01}"
            "{0.99\\times 0.01 + 0.05\\times 0.99} = 0.167$", fontsize=13, va="top")
    ax.text(0, .36,
            "Xét nghiệm chính xác 99% NHƯNG bệnh chỉ\n"
            "gặp ở 1% dân số → prior $P(\\mathrm{bệnh})$ quá nhỏ\n"
            "khiến posterior vẫn thấp.\n\n"
            "Đây chính là lý do Naive Bayes LUÔN nhân\n"
            "likelihood với PRIOR — bỏ prior đi là sai\n"
            "hoàn toàn ở các bài mất cân bằng.",
            fontsize=9.3, va="top", color="#374151")
    fig.suptitle("Định lý Bayes: prior × likelihood → posterior", fontweight="bold")
    fig.tight_layout()
    save(fig, "01_dinh_ly_bayes.png")


# ---------------------------------------------------------------- 2
def fig_naive_assumption():
    rng = np.random.default_rng(3)
    n = 800
    # feature tương quan mạnh
    m = rng.multivariate_normal([0, 0], [[1, .88], [.88, 1]], n)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.1))

    ax = axes[0]
    ax.scatter(m[:, 0], m[:, 1], s=9, alpha=.35, color=C1)
    ax.set_title("Phân phối THẬT $P(x_1,x_2\\,|\\,y)$\n(hai feature tương quan $\\rho=0.88$)",
                 fontsize=10)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_aspect("equal")
    ax.set_xlim(-3.6, 3.6); ax.set_ylim(-3.6, 3.6)

    ax = axes[1]
    fake = np.c_[m[:, 0], rng.permutation(m[:, 1])]     # phá tương quan = giả định naive
    ax.scatter(fake[:, 0], fake[:, 1], s=9, alpha=.35, color=C2)
    ax.set_title("Naive Bayes NGHĨ phân phối là\n$P(x_1|y)\\cdot P(x_2|y)$ (đã mất tương quan)",
                 fontsize=10)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_aspect("equal")
    ax.set_xlim(-3.6, 3.6); ax.set_ylim(-3.6, 3.6)

    ax = axes[2]; ax.axis("off")
    ax.text(0, 1.02, "Giả định naive đánh đổi cái gì?", fontsize=11.5,
            fontweight="bold", va="top")
    ax.text(0, .92, "$P(x_1,\\dots,x_n\\mid y)=\\prod_{i=1}^{n} P(x_i\\mid y)$",
            fontsize=12.5, va="top")
    ax.text(0, .72,
            "MẤT: toàn bộ thông tin về tương quan giữa\ncác feature (hình giữa méo hẳn so với trái).",
            fontsize=9.2, va="top", color="#374151")
    ax.text(0, .57,
            "ĐƯỢC: số tham số phải ước lượng giảm từ\nhàm MŨ xuống hàm TUYẾN TÍNH.",
            fontsize=9.2, va="top", color="#374151")
    ax.text(0, .42, "Với $n$ feature nhị phân, mỗi lớp cần:", fontsize=9.4,
            va="top", fontweight="bold")
    ax.text(0, .34,
            "• Phân phối liên kết đầy đủ: $2^n-1$ tham số\n"
            "    $n=30$ → hơn 1 TỶ tham số\n"
            "• Naive Bayes: chỉ $n$ tham số\n"
            "    $n=30$ → đúng 30",
            fontsize=9.2, va="top", color="#374151")
    ax.text(0, .08,
            "Sai giả định nhưng vẫn dùng được, vì phân loại\n"
            "chỉ cần argmax đúng — không cần xác suất đúng.",
            fontsize=9.0, va="top", color=C2, style="italic")
    fig.suptitle("Vì sao gọi là \"NAIVE\": đánh đổi độ chính xác của xác suất lấy sự đơn giản",
                 fontweight="bold")
    fig.tight_layout()
    save(fig, "02_gia_dinh_naive.png")


# ---------------------------------------------------------------- 3
def fig_three_variants():
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 3.9))

    ax = axes[0]
    w = .34
    ax.bar([1 - w/2, 2 - w/2], [.25, .75], width=w, color=C1, label="lớp spam")
    ax.bar([1 + w/2, 2 + w/2], [.80, .20], width=w, color=C3, label="lớp ham")
    ax.set_xticks([1, 2]); ax.set_xticklabels(["$x_i=0$\n(không có từ)", "$x_i=1$\n(có từ)"])
    ax.set_ylabel("$P(x_i\\,|\\,y)$")
    ax.legend(fontsize=8.5, loc="center", bbox_to_anchor=(.5, .42), framealpha=.94)
    ax.set_title("BernoulliNB — feature NHỊ PHÂN\n"
                 "$P(x_i|y)=p^{x_i}(1-p)^{1-x_i}$", fontsize=10)
    ax.set_ylim(0, 1.18)
    ax.text(.02, .97, "Chú ý: với Bernoulli, việc từ KHÔNG xuất hiện\ncũng được tính là bằng chứng",
            transform=ax.transAxes, fontsize=8.4, color="#374151", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=.93,
                      edgecolor="0.8"))

    ax = axes[1]
    from math import factorial
    k = np.arange(0, 9)
    for lam, col, lbl in [(1.2, C1, "lớp spam"), (3.4, C3, "lớp ham")]:
        pk = np.exp(-lam) * lam ** k / np.array([factorial(int(i)) for i in k])
        ax.bar(k + (.18 if col == C3 else -.18), pk, width=.36, color=col, label=lbl)
    ax.set_xlabel("$x_i$ = số lần từ xuất hiện"); ax.set_ylabel("$P(x_i\\,|\\,y)$")
    ax.legend(fontsize=8.5, loc="center right", framealpha=.94)
    ax.set_title("MultinomialNB — feature ĐẾM\n"
                 "$P(x|y)\\propto\\prod_i p_{i,y}^{x_i}$", fontsize=10)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.30)
    ax.text(.98, .97, "Chỉ đếm những từ CÓ mặt;\ntừ vắng mặt không đóng góp gì",
            transform=ax.transAxes, fontsize=8.4, color="#374151", va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=.93,
                      edgecolor="0.8"))

    ax = axes[2]
    xs = np.linspace(-4, 9, 500)
    for mu, sd, col, lbl in [(1.0, 1.0, C1, "lớp spam: $\\mu$=1.0, $\\sigma$=1.0"),
                             (4.5, 1.6, C3, "lớp ham: $\\mu$=4.5, $\\sigma$=1.6")]:
        pdf = np.exp(-(xs - mu) ** 2 / (2 * sd ** 2)) / (sd * np.sqrt(2 * np.pi))
        ax.plot(xs, pdf, color=col, lw=2.3, label=lbl); ax.fill_between(xs, pdf, alpha=.2, color=col)
    ax.set_xlabel("$x_i$ (giá trị liên tục)"); ax.set_ylabel("mật độ $P(x_i\\,|\\,y)$")
    ax.set_ylim(0, .49)
    ax.legend(fontsize=8.2, loc="upper right", framealpha=.94)
    ax.set_title("GaussianNB — feature LIÊN TỤC\n"
                 "mỗi (feature, lớp) học riêng $\\mu$ và $\\sigma$", fontsize=10)
    fig.suptitle("Ba biến thể Naive Bayes chỉ khác nhau ở cách mô hình hoá $P(x_i\\,|\\,y)$",
                 fontweight="bold")
    fig.tight_layout()
    save(fig, "03_ba_bien_the.png")


# ---------------------------------------------------------------- 4
def fig_gaussian_nb_boundary():
    from matplotlib.patches import Ellipse
    from sklearn.naive_bayes import GaussianNB
    from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                               QuadraticDiscriminantAnalysis)
    rng = np.random.default_rng(1)
    n = 220
    A = rng.multivariate_normal([-1.1, -0.4], [[1.0, .75], [.75, 1.0]], n)
    B = rng.multivariate_normal([1.6, 1.2], [[1.6, -.9], [-.9, 1.0]], n)
    X = np.vstack([A, B]); y = np.r_[np.zeros(n), np.ones(n)]
    xx, yy = np.meshgrid(np.linspace(-5, 6, 320), np.linspace(-4.5, 5.5, 320))
    grid = np.c_[xx.ravel(), yy.ravel()]

    def draw_ellipse(ax, mean, cov, color):
        """Vẽ ellipse 2-sigma của một phân phối Gaussian 2 chiều."""
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        ang = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        for k in (1, 2):
            ax.add_patch(Ellipse(mean, 2 * k * np.sqrt(vals[0]), 2 * k * np.sqrt(vals[1]),
                                 angle=ang, fill=False, edgecolor=color, lw=1.8,
                                 ls="--", zorder=6))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    specs = [
        (GaussianNB(), "GaussianNB",
         "giả định 2 feature ĐỘC LẬP → hiệp phương sai\nbị ép về dạng đường chéo → ellipse SONG SONG TRỤC"),
        (QuadraticDiscriminantAnalysis(store_covariance=True), "QDA",
         "hiệp phương sai ĐẦY ĐỦ, mỗi lớp một ma trận\n→ ellipse NGHIÊNG theo đúng dáng dữ liệu"),
        (LinearDiscriminantAnalysis(store_covariance=True), "LDA",
         "hiệp phương sai đầy đủ nhưng CHUNG cho mọi lớp\n→ hai ellipse giống hệt nhau, ranh giới THẲNG"),
    ]
    for ax, (mdl, name, sub) in zip(axes, specs):
        m = mdl.fit(X, y)
        Z = m.predict_proba(grid)[:, 1].reshape(xx.shape)
        ax.contourf(xx, yy, Z, levels=np.linspace(0, 1, 21), cmap="RdBu_r", alpha=.55)
        ax.contour(xx, yy, Z, levels=[.5], colors="k", linewidths=2.4)
        ax.scatter(A[:, 0], A[:, 1], s=10, color="#1e3a8a", alpha=.55, zorder=4)
        ax.scatter(B[:, 0], B[:, 1], s=10, color="#7f1d1d", alpha=.55, marker="s", zorder=4)

        if name == "GaussianNB":
            covs = [np.diag(v) for v in m.var_]; means = m.theta_
        elif name == "QDA":
            covs = list(m.covariance_); means = m.means_
        else:
            covs = [m.covariance_, m.covariance_]; means = m.means_
        for mu, cv, col in zip(means, covs, ["#1e3a8a", "#7f1d1d"]):
            draw_ellipse(ax, mu, cv, col)

        ax.set_xlim(-5, 6); ax.set_ylim(-4.5, 5.5)
        ax.set_title(f"{name}   —   acc = {m.score(X, y) * 100:.1f}%",
                     fontsize=10.5, fontweight="bold")
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
        ax.text(.02, .03, sub, transform=ax.transAxes, fontsize=8.3, va="bottom",
                bbox=dict(boxstyle="round,pad=0.32", facecolor="white",
                          alpha=.94, edgecolor="0.8"))
    fig.suptitle("Đường nét đứt = ellipse Gaussian mà mỗi model THỰC SỰ khớp vào dữ liệu "
                 "(mức 1σ và 2σ)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, .94])
    save(fig, "04_gaussian_nb_bien_quyet_dinh.png")


# ---------------------------------------------------------------- 5
def fig_laplace_smoothing():
    counts = np.array([12, 7, 3, 0, 0, 1])
    words = ["good", "great", "love", "sublime", "zesty", "nice"]
    alphas = [0.0, 0.01, 1.0, 100.0]
    V = len(words)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4))
    ax = axes[0]
    w = .2
    for j, a in enumerate(alphas):
        p = (counts + a) / (counts.sum() + a * V)
        ax.bar(np.arange(V) + (j - 1.5) * w, p, width=w, label=f"$\\alpha$ = {a}")
    ax.set_xticks(range(V)); ax.set_xticklabels(words, rotation=25, ha="right")
    ax.set_ylabel("$P(\\mathrm{từ}\\,|\\,\\mathrm{lớp\\ positive})$"); ax.legend(fontsize=8)
    ax.set_title("$\\alpha$=0: hai từ có xác suất = 0\n"
                 "→ một từ lạ đủ giết cả tích", fontsize=10)
    ax.annotate("$P=0$ !", (3, 0), xytext=(3.1, .18), color=C2, fontsize=10,
                arrowprops=dict(arrowstyle="->", color=C2))

    ax = axes[1]
    ags = np.geomspace(1e-3, 1e3, 200)
    for i, c in enumerate(counts):
        ax.plot(ags, (c + ags) / (counts.sum() + ags * V), lw=1.9,
                label=f"{words[i]} (đếm={c})")
    ax.axhline(1 / V, color="k", ls="--", lw=1.4)
    ax.text(1.5e-3, 1 / V * 1.08, "$1/V$ — phân phối đều", fontsize=8.3)
    ax.set_xscale("log"); ax.set_xlabel(r"$\alpha$"); ax.set_ylabel("xác suất ước lượng")
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("$\\alpha$ càng lớn, mọi từ càng bị kéo\nvề phân phối ĐỀU (quên dữ liệu)",
                 fontsize=10)

    ax = axes[2]; ax.axis("off")
    ax.text(0, 1.02, "Laplace / Lidstone smoothing", fontsize=11.5,
            fontweight="bold", va="top")
    ax.text(0, .90, r"$\hat{P}(x_i \mid y)=\dfrac{N_{i,y}+\alpha}{N_y + \alpha V}$",
            fontsize=14, va="top")
    ax.text(0, .68,
            "$N_{i,y}$: số lần từ $i$ xuất hiện trong lớp $y$\n"
            "$N_y$: tổng số từ của lớp $y$\n"
            "$V$: kích thước từ điển\n"
            "$\\alpha$: lượng \"đếm ảo\" cộng thêm",
            fontsize=9.0, va="top", color="#374151")
    ax.text(0, .42,
            "• $\\alpha=1$: Laplace (add-one) — mặc định sklearn\n"
            "• $0<\\alpha<1$: Lidstone — thường tốt hơn cho text\n"
            "• $\\alpha\\to\\infty$: mọi từ như nhau → model vô dụng\n"
            "• $\\alpha=0$: từ chưa từng thấy → $P=0$ → sập",
            fontsize=9.0, va="top", color="#374151")
    ax.text(0, .13,
            "Cách đọc theo Bayes: $\\alpha$ chính là prior\nDirichlet trên phân phối từ.",
            fontsize=8.9, va="top", style="italic", color=C1)
    fig.suptitle("Laplace smoothing chữa bài toán \"xác suất bằng 0\"", fontweight="bold")
    fig.tight_layout()
    save(fig, "05_laplace_smoothing.png")


# ---------------------------------------------------------------- 6
def fig_log_probabilities():
    rng = np.random.default_rng(0)
    p = rng.uniform(.001, .05, 400)
    prod = np.cumprod(p)
    logsum = np.cumsum(np.log(p))

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9))
    ax = axes[0]
    ax.plot(prod, color=C2, lw=2.2)
    ax.axhline(np.finfo(float).tiny, color="k", ls="--", lw=1.4)
    ax.text(210, np.finfo(float).tiny * 60,
            "giới hạn dưới của float64\n($\\approx 2.2\\times10^{-308}$)",
            fontsize=8.4, va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=.93,
                      edgecolor="0.8"))
    z = np.argmax(prod == 0) if (prod == 0).any() else len(prod)
    ax.axvline(z, color=C2, ls=":", lw=1.8)
    ax.text(z + 12, 1e-120, f"về ĐÚNG 0 sau {z} từ\n→ mọi lớp đều bằng 0\n→ không so sánh được",
            color=C2, fontsize=8.6, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=.93,
                      edgecolor="0.8"))
    ax.set_yscale("log"); ax.set_ylim(1e-320, 1)
    ax.set_xlabel("số từ đã nhân"); ax.set_ylabel(r"$\prod P(x_i|y)$")
    ax.set_title("Nhân trực tiếp: TRÀN SỐ DƯỚI (underflow)", fontsize=10)

    ax = axes[1]
    ax.plot(logsum, color=C3, lw=2.2)
    ax.set_xlabel("số từ đã cộng"); ax.set_ylabel(r"$\sum \log P(x_i|y)$")
    ax.set_title("Cộng log: tuyến tính, an toàn tuyệt đối", fontsize=10)
    ax.set_ylim(logsum[-1] * 1.06, abs(logsum[-1]) * .30)
    ax.text(.97, .96,
            "$\\arg\\max$ không đổi vì $\\log$ đơn điệu tăng:\n"
            "$\\arg\\max_y \\prod_i P = \\arg\\max_y \\sum_i \\log P$",
            transform=ax.transAxes, fontsize=9, color="#374151", va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.32", facecolor="white", alpha=.93,
                      edgecolor="0.8"))
    fig.suptitle("Vì sao mọi thư viện Naive Bayes làm việc trên thang LOG", fontweight="bold")
    fig.tight_layout()
    save(fig, "06_lam_viec_tren_thang_log.png")


# ---------------------------------------------------------------- 7
def fig_generative_vs_discriminative():
    """Đường học của NB và Logistic trong HAI thế giới: giả định độc lập đúng và sai."""
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    def curve(rho, d=50, n=9000, sep=.30, seed=0):
        """rho = 0: feature độc lập có điều kiện (đúng giả định NB).
           rho > 0: có yếu tố chung -> feature tương quan (sai giả định NB)."""
        rng = np.random.default_rng(seed)
        y = rng.integers(0, 2, n)
        mu = rng.normal(0, sep, (2, d))
        shared = rng.normal(0, 1, (n, 1))
        X = mu[y] + np.sqrt(rho) * shared + np.sqrt(1 - rho) * rng.normal(0, 1, (n, d))
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.45, random_state=1, stratify=y)
        sizes = [16, 32, 64, 128, 256, 512, 1024, 2048, 4000]
        nb, lr = [], []
        r = np.random.default_rng(seed + 1)
        for sz in sizes:
            a, b = [], []
            for _ in range(20):
                idx = r.choice(len(Xtr), sz, replace=False)
                if len(np.unique(ytr[idx])) < 2:
                    continue
                a.append(GaussianNB().fit(Xtr[idx], ytr[idx]).score(Xte, yte))
                b.append(LogisticRegression(max_iter=4000).fit(Xtr[idx], ytr[idx]).score(Xte, yte))
            nb.append(np.mean(a)); lr.append(np.mean(b))
        return sizes, np.array(nb), np.array(lr)

    fig = plt.figure(figsize=(15, 4.5))
    panels = [
        (0.0, "THẾ GIỚI A — giả định độc lập ĐÚNG\n(feature độc lập có điều kiện)",
         "NB đúng mô hình → hội tụ sớm (từ ~100 mẫu)\nvà về đích CAO HƠN Logistic"),
        (0.55, "THẾ GIỚI B — giả định độc lập SAI\n(các feature chia sẻ một yếu tố chung)",
         "NB bị CHẶN bởi giả định sai;\nLogistic vượt lên và giữ khoảng cách"),
    ]
    for k, (rho, ttl, note) in enumerate(panels):
        sizes, nb, lr = curve(rho)
        ax = fig.add_subplot(1, 3, k + 1)
        ax.plot(sizes, nb, "o-", color=C3, lw=2.3, label="Gaussian NB (sinh mẫu)")
        ax.plot(sizes, lr, "s-", color=C1, lw=2.3, label="Logistic Regression (phân biệt)")
        ax.set_xscale("log")
        ax.set_xlabel("số mẫu huấn luyện (thang log)")
        ax.set_ylabel("accuracy trên tập test")
        ax.legend(fontsize=8.2, loc="lower right")
        ax.set_title(ttl, fontsize=9.8)
        hi = max(nb.max(), lr.max()); lo = min(nb.min(), lr.min())
        ax.set_ylim(lo - (hi - lo) * .08, hi + (hi - lo) * .28)
        ax.text(sizes[0] * 1.05, hi + (hi - lo) * .22, note, fontsize=8.6,
                color="#374151", va="top")

    ax = fig.add_subplot(1, 3, 3); ax.axis("off")
    rows = [
        ["", "Sinh mẫu\n(Generative)", "Phân biệt\n(Discriminative)"],
        ["Học cái gì", "$P(x,y)$\ntoàn bộ thế giới", "$P(y\\mid x)$\nchỉ ranh giới"],
        ["Ví dụ", "Naive Bayes, LDA,\nGMM, HMM", "Logistic, SVM,\ncây, MLP"],
        ["Tốc độ train", "Một lượt đếm", "Tối ưu lặp"],
        ["Cần bao nhiêu mẫu\nđể hội tụ", "Rất ít", "Nhiều hơn"],
        ["Trần hiệu năng", "Bị chặn nếu\ngiả định sai", "Cao hơn khi\nđủ dữ liệu"],
        ["Sinh dữ liệu mới", "Có thể", "Không thể"],
        ["Dữ liệu thiếu", "Xử lý tự nhiên", "Phải điền khuyết"],
    ]
    t = ax.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="left",
                 colWidths=[.30, .35, .35])
    t.auto_set_font_size(False); t.set_fontsize(8.0); t.scale(1, 2.35)
    for j in range(3):
        t[0, j].set_facecolor("#e5e7eb"); t[0, j].set_text_props(fontweight="bold")
    ax.set_title("Hai trường phái mô hình hoá", fontsize=10.5, fontweight="bold", pad=16)

    fig.suptitle("Naive Bayes là mô hình SINH MẪU: bias cao, variance thấp — "
                 "lợi hay hại tuỳ giả định có đúng không", fontweight="bold")
    fig.tight_layout()
    save(fig, "07_sinh_mau_vs_phan_biet.png")


# ---------------------------------------------------------------- 8
def fig_correlation_hurts():
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    rng = np.random.default_rng(7)
    n = 600
    base = rng.normal(size=(n, 2))
    y = (base[:, 0] + base[:, 1] > 0).astype(int)
    X0 = base + rng.normal(0, .55, base.shape)

    reps = [1, 2, 4, 8, 16]
    nb, lr = [], []
    for r in reps:
        X = np.hstack([X0] + [X0[:, :1] + rng.normal(0, .02, (n, 1)) for _ in range(r - 1)])
        nb.append(cross_val_score(GaussianNB(), X, y, cv=5).mean())
        lr.append(cross_val_score(LogisticRegression(max_iter=2000), X, y, cv=5).mean())

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4))
    ax = axes[0]
    ax.plot(reps, nb, "o-", color=C3, lw=2.2, label="Gaussian NB")
    ax.plot(reps, lr, "s-", color=C1, lw=2.2, label="Logistic Regression")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("số bản SAO của cùng một feature (thông tin không hề tăng)")
    ax.set_ylabel("accuracy (CV 5-fold)"); ax.legend(fontsize=8.5)
    ax.set_title("Nhân bản feature → NB tụt, Logistic gần như đứng yên", fontsize=10)

    ax = axes[1]; ax.axis("off")
    ax.text(0, 1.00, "Vì sao nhân bản feature lại giết Naive Bayes?", fontsize=11,
            fontweight="bold", va="top")
    ax.text(0, .88,
            "Giả sử $x_1$ được sao thành $x_1, x_1', x_1''$ (giống hệt nhau).\n"
            "Naive Bayes nhân likelihood ba lần:",
            fontsize=9.3, va="top", color="#374151")
    ax.text(0, .70, r"$P(x_1|y)\cdot P(x_1'|y)\cdot P(x_1''|y) = P(x_1|y)^3$",
            fontsize=12.5, va="top")
    ax.text(0, .56,
            "→ MỘT bằng chứng bị TÍNH BA LẦN. Model trở nên\n"
            "tự tin thái quá và lệch hẳn về phía $x_1$ ủng hộ.",
            fontsize=9.3, va="top", color="#374151")
    ax.text(0, .40,
            "Logistic Regression thì chia trọng số giữa các bản\n"
            "sao → tổng đóng góp không đổi.",
            fontsize=9.3, va="top", color="#374151")
    ax.text(0, .22,
            "THỰC HÀNH: trước khi dùng NB, hãy bỏ bớt feature\n"
            "trùng lặp (kiểm tra ma trận tương quan). Với text,\n"
            "dùng TF-IDF thay raw count cũng giảm đếm trùng.",
            fontsize=9.0, va="top", color=C2, style="italic")
    fig.suptitle("Điểm yếu chí mạng: feature TRÙNG LẶP bị đếm nhiều lần", fontweight="bold")
    fig.tight_layout()
    save(fig, "08_tuong_quan_pha_naive_bayes.png")


# ---------------------------------------------------------------- 9
def fig_text_pipeline():
    docs = ["the course is great", "great teacher great course", "the course is bad"]
    labels = ["positive", "positive", "negative"]
    vocab = ["bad", "course", "great", "is", "teacher", "the"]
    counts = np.array([[0, 1, 1, 1, 0, 1],
                       [0, 1, 2, 0, 1, 0],
                       [1, 1, 0, 1, 0, 1]])

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 3.9))
    ax = axes[0]; ax.axis("off")
    ax.text(0, 1.00, "1. Văn bản thô", fontsize=11, fontweight="bold", va="top")
    for i, (d, l) in enumerate(zip(docs, labels)):
        col = C3 if l == "positive" else C2
        ax.text(0, .82 - i * .20, f'doc {i}: "{d}"', fontsize=9.4, va="top")
        ax.text(0, .74 - i * .20, f"nhãn: {l}", fontsize=8.8, va="top", color=col)
    ax.text(0, .20, "CountVectorizer tách từ, hạ chữ thường,\nbỏ stop-words, dựng từ điển.",
            fontsize=8.6, va="top", color="dimgray")
    ax = axes[1]
    im = ax.imshow(counts, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(vocab))); ax.set_xticklabels(vocab, rotation=35, ha="right")
    ax.set_yticks(range(3)); ax.set_yticklabels([f"doc {i}" for i in range(3)])
    ax.grid(False)
    for i in range(3):
        for j in range(len(vocab)):
            ax.text(j, i, counts[i, j], ha="center", va="center", fontsize=10,
                    color="white" if counts[i, j] > 1 else "black")
    ax.set_title("2. Ma trận Bag-of-Words\n(hàng = văn bản, cột = từ)", fontsize=10)

    ax = axes[2]; ax.axis("off")
    ax.text(0, 1.02, "3. Naive Bayes đếm và tính", fontsize=11, fontweight="bold", va="top")
    ax.text(0, .90, r"$P(y)=\dfrac{\text{số doc lớp } y}{\text{tổng doc}}$",
            fontsize=12, va="top")
    ax.text(0, .68,
            r"$P(w \mid y)=\dfrac{\text{đếm}(w,y)+\alpha}"
            r"{\sum_{w'}\text{đếm}(w',y)+\alpha V}$", fontsize=12, va="top")
    ax.text(0, .44, "Dự đoán doc mới:", fontsize=9.4, va="top", fontweight="bold")
    ax.text(0, .36,
            r"$\hat{y}=\arg\max_y [\log P(y) + \sum_w n_w \log P(w\mid y)]$",
            fontsize=10, va="top")
    ax.text(0, .21,
            "Toàn bộ \"huấn luyện\" chỉ là ĐẾM — một lượt duyệt\n"
            "dữ liệu, không lặp, không learning rate. Đó là lý do\n"
            "NB nhanh hơn mọi model khác hàng trăm lần.",
            fontsize=8.9, va="top", color="#374151")
    ax.text(0, .01, "⚠️ fit_transform CHỈ trên train, transform trên test.",
            fontsize=8.9, va="top", color=C2, fontweight="bold")
    fig.suptitle("Đường ống phân loại văn bản với Naive Bayes", fontweight="bold")
    fig.tight_layout()
    save(fig, "09_duong_ong_van_ban.png")


if __name__ == "__main__":
    fig_bayes_theorem()
    fig_naive_assumption()
    fig_three_variants()
    fig_gaussian_nb_boundary()
    fig_laplace_smoothing()
    fig_log_probabilities()
    fig_generative_vs_discriminative()
    fig_correlation_hurts()
    fig_text_pipeline()
    print("Xong.")
