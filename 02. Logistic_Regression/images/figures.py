"""
Sinh toàn bộ hình minh hoạ cho Lab 02 - Logistic Regression.

Chạy:  python figures.py
Kết quả: các file .png trong cùng thư mục, được nhúng vào notebook bằng markdown.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


# ---------------------------------------------------------------- 1
def fig_why_not_linear():
    from sklearn.linear_model import LinearRegression, LogisticRegression
    rng = np.random.default_rng(1)
    x = np.r_[rng.normal(2.2, .8, 20), rng.normal(6.0, .8, 20)]
    y = np.r_[np.zeros(20), np.ones(20)]
    xo = np.r_[x, [17., 18., 19.]]; yo = np.r_[y, [1., 1., 1.]]
    xs = np.linspace(-1, 20, 400).reshape(-1, 1)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 3.9))
    for ax, (xa, ya, ttl) in zip(axes[:2],
            [(x, y, "Linear Regression trên nhãn 0/1"),
             (xo, yo, "…và chỉ 3 điểm ở xa đã phá nát ranh giới")]):
        lin = LinearRegression().fit(xa.reshape(-1, 1), ya)
        pr = lin.predict(xs).ravel()
        ax.scatter(xa, ya, s=42, c=["#93c5fd" if t == 0 else "#fca5a5" for t in ya],
                   edgecolor="k", linewidth=.5, zorder=4)
        ax.plot(xs, pr, color=C1, lw=2.2)
        ax.axhline(.5, color="gray", ls=":", lw=1.4)
        ax.fill_between(xs.ravel(), pr, 1, where=pr > 1, color=C2, alpha=.18)
        ax.fill_between(xs.ravel(), pr, 0, where=pr < 0, color=C2, alpha=.18)
        cut = (.5 - lin.intercept_) / lin.coef_[0]
        ax.axvline(cut, color=C2, ls="--", lw=1.8)
        ax.text(cut + .3, .06, f"ngưỡng\nx={cut:.1f}", color=C2, fontsize=8.5)
        ax.set_ylim(-.55, 1.6); ax.set_title(ttl, fontsize=10)
        ax.set_xlabel("x"); ax.set_ylabel("y / xác suất")
    axes[0].text(-.6, 1.32, "vùng đỏ: dự đoán > 1 hoặc < 0\n→ KHÔNG thể đọc là xác suất",
                 color=C2, fontsize=8.5)

    log = LogisticRegression().fit(xo.reshape(-1, 1), yo)
    ax = axes[2]
    ax.scatter(xo, yo, s=42, c=["#93c5fd" if t == 0 else "#fca5a5" for t in yo],
               edgecolor="k", linewidth=.5, zorder=4)
    ax.plot(xs, log.predict_proba(xs)[:, 1], color=C3, lw=2.4)
    ax.axhline(.5, color="gray", ls=":", lw=1.4)
    cut = -log.intercept_[0] / log.coef_[0][0]
    ax.axvline(cut, color=C3, ls="--", lw=1.8)
    ax.text(cut + .3, .06, f"ngưỡng\nx={cut:.1f}", color=C3, fontsize=8.5)
    ax.set_ylim(-.55, 1.6); ax.set_xlabel("x"); ax.set_ylabel("xác suất")
    ax.set_title("Logistic Regression: cùng dữ liệu có outlier\n→ ngưỡng gần như không đổi", fontsize=10)
    fig.suptitle("Vì sao không dùng Linear Regression cho bài phân loại", fontweight="bold")
    fig.tight_layout()
    save(fig, "01_vi_sao_khong_dung_linear.png")


# ---------------------------------------------------------------- 2
def fig_sigmoid():
    z = np.linspace(-8, 8, 600)
    s = sigmoid(z); ds = s * (1 - s)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(z, s, color=C1, lw=2.6, label=r"$\sigma(z)=\dfrac{1}{1+e^{-z}}$")
    ax.axhline(.5, color="gray", ls=":", lw=1.2); ax.axvline(0, color="gray", ls=":", lw=1.2)
    ax.axhline(1, color="k", lw=.8, alpha=.4); ax.axhline(0, color="k", lw=.8, alpha=.4)
    ax.axvspan(-8, -4, color=C2, alpha=.10); ax.axvspan(4, 8, color=C2, alpha=.10)
    ax.text(-7.8, .72, "vùng BÃO HOÀ\n$\\sigma'\\approx 0$", color=C2, fontsize=8.5)
    ax.text(4.2, .28, "vùng BÃO HOÀ\n$\\sigma'\\approx 0$", color=C2, fontsize=8.5)
    ax.scatter([0], [.5], color=C4, s=70, zorder=5)
    ax.annotate("$\\sigma(0)=0.5$\n(ngưỡng quyết định mặc định)", (0, .5),
                xytext=(-7.6, .18), fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color=C4))
    ax.set_ylim(-.08, 1.12); ax.set_xlabel("$z = w^Tx+b$ (logit)")
    ax.set_ylabel("xác suất $\\hat{p}$"); ax.legend(fontsize=10, loc="lower right")
    ax.set_title("Sigmoid ép mọi số thực về khoảng (0, 1)", fontsize=10.5)

    ax = axes[1]
    ax.plot(z, s, color=C1, lw=1.6, alpha=.5, label=r"$\sigma(z)$")
    ax.plot(z, ds, color=C2, lw=2.6, label=r"$\sigma'(z)=\sigma(z)\,(1-\sigma(z))$")
    ax.axhline(.25, color=C4, ls="--", lw=1.4)
    ax.text(-7.8, .262, "đạo hàm cực đại = 0.25 tại z=0", color=C4, fontsize=8.5)
    ax.annotate("z càng xa 0, gradient càng tắt\n→ gốc rễ của VANISHING GRADIENT\n(gặp lại ở bài MLP)",
                (5, ds[np.argmin(abs(z - 5))]), xytext=(0.4, .155), fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color="dimgray"), color="dimgray")
    ax.set_xlabel("z"); ax.set_ylabel("giá trị"); ax.legend(fontsize=9.5)
    ax.set_title("Đạo hàm sigmoid — đẹp nhưng dễ tắt", fontsize=10.5)
    fig.suptitle("Hàm sigmoid và đạo hàm của nó", fontweight="bold")
    fig.tight_layout()
    save(fig, "02_ham_sigmoid.png")


# ---------------------------------------------------------------- 3
def fig_odds_logit():
    p = np.linspace(.001, .999, 600)
    odds = p / (1 - p)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 3.8))

    ax = axes[0]
    ax.plot(p, odds, color=C1, lw=2.4)
    ax.axhline(1, color="gray", ls=":"); ax.axvline(.5, color="gray", ls=":")
    ax.set_ylim(0, 10); ax.set_xlabel("xác suất $p$"); ax.set_ylabel("odds $=p/(1-p)$")
    ax.set_title("Odds: $[0,1] \\rightarrow [0,\\infty)$", fontsize=10)
    ax.text(.06, 8.2, "p = 0.5 → odds = 1 (hoà)\np = 0.9 → odds = 9 (9 ăn 1)", fontsize=8.5,
            color="dimgray")

    ax = axes[1]
    ax.plot(p, np.log(odds), color=C3, lw=2.4)
    ax.axhline(0, color="gray", ls=":"); ax.axvline(.5, color="gray", ls=":")
    ax.set_xlabel("xác suất $p$"); ax.set_ylabel("logit $=\\ln\\frac{p}{1-p}$")
    ax.set_title("Logit: $[0,1] \\rightarrow (-\\infty,\\infty)$\n(hàm ngược của sigmoid)", fontsize=10)
    ax.text(.05, 4.2, "Logistic Regression thật ra là\nhồi quy TUYẾN TÍNH trên logit:\n"
                      r"$\ln\frac{p}{1-p} = w^Tx+b$", fontsize=8.8, color="dimgray")

    ax = axes[2]
    ws = np.array([0.2, 0.5, 1.0, 2.0])
    for w, col in zip(ws, [C1, C3, C4, C2]):
        ax.plot(np.linspace(-8, 8, 400), sigmoid(w * np.linspace(-8, 8, 400)),
                color=col, lw=2.1, label=f"$\\|w\\|$ = {w}  (OR = $e^w$ = {np.exp(w):.2f})")
    ax.axhline(.5, color="gray", ls=":")
    ax.set_xlabel("x"); ax.set_ylabel("$\\hat{p}$"); ax.legend(fontsize=8)
    ax.set_title("$\\|w\\|$ lớn → đường dốc đứng\n→ model tự tin hơn", fontsize=10)
    fig.suptitle("Odds, logit và ý nghĩa của hệ số: $e^{w_j}$ là ODDS RATIO", fontweight="bold")
    fig.tight_layout()
    save(fig, "03_odds_va_logit.png")


# ---------------------------------------------------------------- 4
def fig_bce_vs_mse():
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4))

    ph = np.linspace(1e-4, 1 - 1e-4, 600)
    ax = axes[0]
    ax.plot(ph, -np.log(ph), color=C3, lw=2.4, label="nhãn thật $y=1$:  $-\\log \\hat{p}$")
    ax.plot(ph, -np.log(1 - ph), color=C2, lw=2.4, label="nhãn thật $y=0$:  $-\\log(1-\\hat{p})$")
    ax.set_ylim(0, 6); ax.set_xlabel("$\\hat{p}$ model dự đoán"); ax.set_ylabel("mất mát")
    ax.legend(fontsize=8.5); ax.set_title("BCE: sai mà TỰ TIN thì bị phạt vô hạn", fontsize=10)
    ax.annotate("y=1 mà model nói $\\hat{p}$=0.01\n→ loss = 4.6 (rất đau)", (.01, 4.6),
                xytext=(.22, 4.9), fontsize=8.3, arrowprops=dict(arrowstyle="->", color="dimgray"))

    # loss landscape 1 tham số: BCE lồi, MSE+sigmoid không lồi
    rng = np.random.default_rng(0)
    xd = np.array([-2.5, -2.0, -1.6, 2.0, 2.4, 3.0, -3.0, 2.8])
    yd = np.array([0, 0, 0, 1, 1, 1, 1, 0])          # có 2 nhãn "nhiễu" cố ý
    ws = np.linspace(-6, 6, 600)
    bce = [np.mean(-(yd * np.log(sigmoid(w * xd) + 1e-12)
                     + (1 - yd) * np.log(1 - sigmoid(w * xd) + 1e-12))) for w in ws]
    mse = [np.mean((sigmoid(w * xd) - yd) ** 2) for w in ws]
    ax = axes[1]
    ax.plot(ws, bce, color=C3, lw=2.4)
    ax.scatter([ws[int(np.argmin(bce))]], [min(bce)], color=C3, s=70, zorder=5)
    ax.set_xlabel("tham số w"); ax.set_ylabel("BCE")
    ax.set_title("BCE + sigmoid: LỒI\n→ một cực tiểu, GD luôn tìm được", fontsize=10)
    ax = axes[2]
    ax.plot(ws, mse, color=C2, lw=2.4)
    ax.set_xlabel("tham số w"); ax.set_ylabel("MSE")
    ax.set_title("MSE + sigmoid: KHÔNG lồi\n→ vùng phẳng rộng, gradient ≈ 0, GD kẹt", fontsize=10)
    ax.annotate("cao nguyên phẳng:\ngradient tắt, train đứng im", (-5, mse[20]),
                xytext=(-5.6, np.max(mse) * .62), fontsize=8.3,
                arrowprops=dict(arrowstyle="->", color="dimgray"), color="dimgray")
    fig.suptitle("Vì sao Logistic Regression dùng Cross-Entropy chứ không dùng MSE",
                 fontweight="bold")
    fig.tight_layout()
    save(fig, "04_bce_vs_mse.png")


# ---------------------------------------------------------------- 5
def fig_decision_boundary():
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                               n_clusters_per_class=1, class_sep=1.3, random_state=4)
    clf = LogisticRegression().fit(X, y)
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 300),
                         np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 300))
    P = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax = axes[0]
    cf = ax.contourf(xx, yy, P, levels=np.linspace(0, 1, 21), cmap="RdBu_r", alpha=.75)
    cs = ax.contour(xx, yy, P, levels=[.1, .25, .5, .75, .9], colors="k",
                    linewidths=[.8, .8, 2.4, .8, .8])
    ax.clabel(cs, fmt="%.2f", fontsize=7.5)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu_r", s=24, edgecolor="k", linewidth=.4)
    ax.set_title("Ranh giới quyết định là ĐƯỜNG THẲNG $w^Tx+b=0$\n"
                 "(nơi $\\hat{p}=0.5$); các đường khác là đường đồng mức xác suất", fontsize=10)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    plt.colorbar(cf, ax=ax, label="$\\hat{p}(y=1\\,|\\,x)$")
    w = clf.coef_[0]
    ax.arrow(0, -clf.intercept_[0] / w[1], w[0] * .55, w[1] * .55, width=.06,
             color="lime", ec="k", zorder=6, length_includes_head=True)
    ax.text(w[0] * .6, -clf.intercept_[0] / w[1] + w[1] * .62, " $w$", color="k",
            fontsize=11, fontweight="bold")

    # lát cắt theo hướng w
    t = (X @ w + clf.intercept_[0]) / np.linalg.norm(w)
    ts = np.linspace(t.min() - .6, t.max() + .6, 400)
    ax = axes[1]
    ax.plot(ts, sigmoid(np.linalg.norm(w) * ts), color=C1, lw=2.6)
    ax.scatter(t, y, c=y, cmap="RdBu_r", s=26, edgecolor="k", linewidth=.4, zorder=4)
    ax.axhline(.5, color="gray", ls=":"); ax.axvline(0, color="k", ls="--", lw=1.6)
    ax.text(.1, .06, "ranh giới\nquyết định", fontsize=8.5)
    ax.set_xlabel("khoảng cách có dấu tới ranh giới")
    ax.set_ylabel("$\\hat{p}$"); ax.set_ylim(-.12, 1.12)
    ax.set_title("Chiếu dữ liệu lên hướng $w$:\nbài toán trở về đúng đường sigmoid 1 chiều", fontsize=10)
    fig.suptitle("Hình học của Logistic Regression", fontweight="bold")
    fig.tight_layout()
    save(fig, "05_bien_quyet_dinh.png")


# ---------------------------------------------------------------- 6
def fig_regularization_C():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(2)
    Xa = np.r_[rng.normal([-1.6, 0], .55, (40, 2)), rng.normal([1.6, 0], .55, (40, 2))]
    ya = np.r_[np.zeros(40), np.ones(40)]           # tách tuyến tính hoàn toàn

    Cs = [0.001, 0.1, 100]
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 3.9))
    xx, yy = np.meshgrid(np.linspace(-4, 4, 250), np.linspace(-2.6, 2.6, 250))
    for ax, C in zip(axes[:3], Cs):
        clf = LogisticRegression(C=C).fit(Xa, ya)
        P = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
        ax.contourf(xx, yy, P, levels=np.linspace(0, 1, 21), cmap="RdBu_r", alpha=.8)
        ax.contour(xx, yy, P, levels=[.5], colors="k", linewidths=2)
        ax.scatter(Xa[:, 0], Xa[:, 1], c=ya, cmap="RdBu_r", s=22, edgecolor="k", linewidth=.4)
        ax.set_title(f"C = {C}   →   $\\|w\\|$ = {np.linalg.norm(clf.coef_):.2f}", fontsize=10)
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    axes[0].text(-3.9, 2.1, "phạt MẠNH:\nchuyển tiếp rất mờ", fontsize=8.3, color="k")
    axes[2].text(-3.9, 2.1, "phạt YẾU:\nchuyển tiếp gắt như bậc thang", fontsize=8.3, color="k")

    Cgrid = np.geomspace(1e-4, 1e4, 40)
    norms = [np.linalg.norm(LogisticRegression(C=c).fit(Xa, ya).coef_) for c in Cgrid]
    ax = axes[3]
    ax.plot(Cgrid, norms, "o-", color=C2, ms=3.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("C  (= $1/\\alpha$, càng lớn càng ÍT phạt)")
    ax.set_ylabel("$\\|w\\|$")
    ax.set_title("Dữ liệu tách hoàn toàn: bỏ regularization\nthì $\\|w\\|$ chạy ra vô cực", fontsize=10)
    fig.suptitle("Tham số C của Logistic Regression: C nhỏ = phạt mạnh = hệ số nhỏ",
                 fontweight="bold")
    fig.tight_layout()
    save(fig, "06_regularization_C.png")


# ---------------------------------------------------------------- 7
def fig_polynomial_logistic():
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_moons
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    X, y = make_moons(n_samples=250, noise=.22, random_state=42)
    xx, yy = np.meshgrid(np.linspace(-2, 3, 300), np.linspace(-1.6, 2, 300))
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 3.9))
    for ax, d in zip(axes, [1, 2, 3, 12]):
        m = make_pipeline(PolynomialFeatures(d), StandardScaler(),
                          LogisticRegression(max_iter=5000)).fit(X, y)
        P = m.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
        ax.contourf(xx, yy, P, levels=np.linspace(0, 1, 21), cmap="RdBu_r", alpha=.75)
        ax.contour(xx, yy, P, levels=[.5], colors="k", linewidths=2)
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu_r", s=20, edgecolor="k", linewidth=.4)
        note = {1: "thẳng — underfit", 2: "cong nhẹ", 3: "vừa đẹp",
                12: "uốn éo — overfit"}[d]
        ax.set_title(f"degree = {d}  ({note})\ntrain acc = {m.score(X, y)*100:.1f}%", fontsize=10)
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    fig.suptitle("Logistic Regression vẫn học được ranh giới CONG — nhờ polynomial features",
                 fontweight="bold")
    fig.tight_layout()
    save(fig, "07_polynomial_logistic.png")


# ---------------------------------------------------------------- 8
def fig_softmax():
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=300, centers=3, cluster_std=1.15, random_state=7)
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1.5, X[:, 0].max() + 1.5, 320),
                         np.linspace(X[:, 1].min() - 1.5, X[:, 1].max() + 1.5, 320))
    grid = np.c_[xx.ravel(), yy.ravel()]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    z = np.linspace(-3, 3, 400)
    ax = axes[0]
    logits = np.vstack([1.4 * z, .2 * z + .6, -1.1 * z + .2])
    e = np.exp(logits - logits.max(0)); sm = e / e.sum(0)
    for i, col in enumerate([C1, C3, C2]):
        ax.plot(z, sm[i], color=col, lw=2.3, label=f"$\\hat{{p}}$(lớp {i})")
    ax.set_xlabel("một lát cắt của không gian đặc trưng"); ax.set_ylabel("xác suất")
    ax.legend(fontsize=8.5)
    ax.set_title("Softmax: 3 xác suất luôn CỘNG LẠI BẰNG 1\n"
                 r"$\hat{p}_c=e^{z_c}/\sum_k e^{z_k}$", fontsize=10)

    from sklearn.multiclass import OneVsRestClassifier
    models = [
        (LogisticRegression(max_iter=2000),
         "multinomial (softmax thật)\nmột bài toán tối ưu chung cho 3 lớp"),
        (OneVsRestClassifier(LogisticRegression(max_iter=2000)),
         "one-vs-rest\n3 bộ nhị phân riêng rồi so điểm"),
    ]
    for ax, (est, ttl) in zip(axes[1:], models):
        clf = est.fit(X, y)
        Z = clf.predict(grid).reshape(xx.shape)
        ax.contourf(xx, yy, Z, levels=[-.5, .5, 1.5, 2.5], colors=["#bfdbfe", "#bbf7d0", "#fecaca"])
        ax.contour(xx, yy, Z, levels=[.5, 1.5], colors="k", linewidths=1.6)
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="brg", s=20, edgecolor="k", linewidth=.35)
        ax.set_title(f"{ttl}\nacc = {clf.score(X, y)*100:.1f}%", fontsize=9.5)
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    fig.suptitle("Mở rộng lên nhiều lớp: Softmax Regression", fontweight="bold")
    fig.tight_layout()
    save(fig, "08_softmax_da_lop.png")


# ---------------------------------------------------------------- 9
def fig_threshold():
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc
    X, y = make_classification(n_samples=1200, n_features=8, n_informative=4,
                               weights=[.75, .25], random_state=3)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.35, random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    pr = clf.predict_proba(Xte)[:, 1]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4)) 
    ax = axes[0]
    ax.hist(pr[yte == 0], bins=40, alpha=.65, color=C1, label="lớp thật = 0")
    ax.hist(pr[yte == 1], bins=40, alpha=.65, color=C2, label="lớp thật = 1")
    for t, col in [(.3, C3), (.5, "k"), (.7, C4)]:
        ax.axvline(t, color=col, ls="--", lw=1.9)
        ax.text(t, ax.get_ylim()[1] * .93, f" {t}", color=col, fontsize=9)
    ax.set_xlabel("$\\hat{p}$ model xuất ra"); ax.set_ylabel("số mẫu")
    ax.legend(fontsize=8.5); ax.set_title("Ngưỡng chỉ là một đường kẻ dọc\ntrên phân phối điểm số", fontsize=10)

    ts = np.linspace(.02, .98, 120)
    P = [precision_score(yte, pr >= t, zero_division=0) for t in ts]
    R = [recall_score(yte, pr >= t) for t in ts]
    F = [f1_score(yte, pr >= t, zero_division=0) for t in ts]
    ax = axes[1]
    ax.plot(ts, P, color=C1, lw=2.2, label="Precision")
    ax.plot(ts, R, color=C2, lw=2.2, label="Recall")
    ax.plot(ts, F, color=C3, lw=2.6, label="F1")
    tb = ts[int(np.argmax(F))]
    ax.axvline(tb, color=C3, ls="--", lw=1.6)
    ax.axvline(.5, color="gray", ls=":", lw=1.6)
    ax.text(tb + .015, .08, f"F1 max tại\nngưỡng {tb:.2f}", color=C3, fontsize=8.5)
    ax.set_xlabel("ngưỡng"); ax.set_ylabel("giá trị"); ax.legend(fontsize=8.5)
    ax.set_title("0.5 KHÔNG phải lúc nào cũng tốt nhất", fontsize=10)

    fpr, tpr, thr = roc_curve(yte, pr)
    ax = axes[2]
    ax.plot(fpr, tpr, color=C1, lw=2.4, label=f"ROC (AUC = {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="đoán mò")
    for t, col in [(.3, C3), (.5, "k"), (.7, C4)]:
        i = int(np.argmin(np.abs(thr - t)))
        ax.scatter([fpr[i]], [tpr[i]], color=col, s=75, zorder=5)
        ax.text(fpr[i] + .03, tpr[i] - .06, f"ngưỡng {t}", color=col, fontsize=8.3)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR (Recall)"); ax.legend(fontsize=8.5, loc="lower right")
    ax.set_title("Mỗi ngưỡng = MỘT ĐIỂM trên đường ROC", fontsize=10)
    fig.suptitle("Đổi ngưỡng không train lại model — chỉ đổi cách ĐỌC xác suất",
                 fontweight="bold")
    fig.tight_layout()
    save(fig, "09_nguong_quyet_dinh.png")


if __name__ == "__main__":
    fig_why_not_linear()
    fig_sigmoid()
    fig_odds_logit()
    fig_bce_vs_mse()
    fig_decision_boundary()
    fig_regularization_C()
    fig_polynomial_logistic()
    fig_softmax()
    fig_threshold()
    print("Xong.")
