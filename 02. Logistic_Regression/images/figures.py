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
    outs = np.array([30., 33., 36., 39., 42.])          # 5 điểm lớp 1 ở rất xa
    xo = np.r_[x, outs]; yo = np.r_[y, np.ones(len(outs))]
    xs = np.linspace(-1, 44, 500).reshape(-1, 1)

    def dots(ax, xa, ya, wrong=None):
        cols = ["#93c5fd" if t == 0 else "#fca5a5" for t in ya]
        ax.scatter(xa, ya, s=48, c=cols, edgecolor="k", linewidth=.5, zorder=4)
        if wrong is not None and wrong.any():
            ax.scatter(xa[wrong], ya[wrong], s=150, marker="x", color="k",
                       linewidth=2.2, zorder=6)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.1))

    # (a) linear, không outlier
    lin0 = LinearRegression().fit(x.reshape(-1, 1), y)
    pr0 = lin0.predict(xs).ravel()
    cut0 = (.5 - lin0.intercept_) / lin0.coef_[0]
    ax = axes[0]
    ax.plot(xs, pr0, color=C1, lw=2.4)
    ax.fill_between(xs.ravel(), pr0, 1, where=pr0 > 1, color=C2, alpha=.16)
    ax.fill_between(xs.ravel(), pr0, 0, where=pr0 < 0, color=C2, alpha=.16)
    dots(ax, x, y)
    ax.axhline(.5, color="gray", ls=":", lw=1.4)
    ax.axvline(cut0, color=C2, ls="--", lw=1.8)
    ax.text(cut0 + .6, -.42, f"ngưỡng x={cut0:.1f}", color=C2, fontsize=9)
    ax.text(-.5, 1.42, "vùng tô đỏ: dự đoán $>1$ hoặc $<0$\n→ KHÔNG đọc được là xác suất",
            color=C2, fontsize=8.8)
    ax.set_title("Linear Regression trên nhãn 0/1\n(chưa có outlier: sai 0/40)", fontsize=10)

    # (b) linear, có outlier
    lin = LinearRegression().fit(xo.reshape(-1, 1), yo)
    pr = lin.predict(xs).ravel()
    cut = (.5 - lin.intercept_) / lin.coef_[0]
    wrong = (lin.predict(x.reshape(-1, 1)) >= .5).astype(int) != y
    ax = axes[1]
    ax.plot(xs, pr, color=C1, lw=2.4)
    ax.fill_between(xs.ravel(), pr, 1, where=pr > 1, color=C2, alpha=.16)
    ax.fill_between(xs.ravel(), pr, 0, where=pr < 0, color=C2, alpha=.16)
    dots(ax, xo, yo, np.r_[wrong, np.zeros(len(outs), bool)])
    ax.axhline(.5, color="gray", ls=":", lw=1.4)
    ax.axvline(cut0, color="gray", ls="--", lw=1.5)
    ax.axvline(cut, color=C2, ls="--", lw=1.9)
    ax.annotate("", xy=(cut, -.30), xytext=(cut0, -.30),
                arrowprops=dict(arrowstyle="->", color=C2, lw=1.8))
    ax.text(cut + .8, -.50, f"ngưỡng trôi\n{cut0:.1f} → {cut:.1f}",
            color=C2, fontsize=9)
    ax.text(20, .16, "✗ = điểm bị phân loại SAI\ndù dữ liệu của nó không đổi",
            fontsize=8.8)
    ax.set_title(f"…thêm 5 điểm lớp 1 ở xa\n→ sai {wrong.sum()}/40 điểm ban đầu", fontsize=10)

    # (c) logistic trên CÙNG dữ liệu có outlier
    log0 = LogisticRegression().fit(x.reshape(-1, 1), y)
    lcut0 = -log0.intercept_[0] / log0.coef_[0][0]
    log = LogisticRegression().fit(xo.reshape(-1, 1), yo)
    lcut = -log.intercept_[0] / log.coef_[0][0]
    lwrong = (log.predict_proba(x.reshape(-1, 1))[:, 1] >= .5).astype(int) != y
    ax = axes[2]
    ax.plot(xs, log.predict_proba(xs)[:, 1], color=C3, lw=2.6)
    dots(ax, xo, yo, np.r_[lwrong, np.zeros(len(outs), bool)])
    ax.axhline(.5, color="gray", ls=":", lw=1.4)
    ax.axvline(lcut, color=C3, ls="--", lw=1.9)
    ax.text(lcut + .6, -.42, f"ngưỡng x={lcut:.1f}", color=C3, fontsize=9)
    ax.text(9, 1.30, f"ngưỡng khi CHƯA có outlier: {lcut0:.2f}\n"
                     f"ngưỡng khi CÓ outlier:      {lcut:.2f}\n→ không xê dịch",
            color=C3, fontsize=8.8)
    ax.set_title(f"Logistic Regression, cùng dữ liệu có outlier\n"
                 f"→ sai {lwrong.sum()}/40 điểm ban đầu", fontsize=10)

    for ax in axes:
        ax.set_ylim(-.62, 1.62); ax.set_xlim(-2, 45)
        ax.set_xlabel("x"); ax.set_ylabel("y / xác suất")
    fig.suptitle("Linear Regression trên bài toán phân loại", fontweight="bold")
    fig.tight_layout()
    save(fig, "01_vi_sao_khong_dung_linear.png")


# ---------------------------------------------------------------- 2
def fig_sigmoid():
    z = np.linspace(-8, 8, 600)
    sg = sigmoid(z); ds = sg * (1 - sg)
    BOX = dict(boxstyle="round,pad=0.32", facecolor="white", alpha=.93, edgecolor="0.8")
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.4))

    ax = axes[0]
    ax.axvspan(-8, -4, color=C2, alpha=.09); ax.axvspan(4, 8, color=C2, alpha=.09)
    ax.plot(z, sg, color=C1, lw=2.8, zorder=4,
            label=r"$\sigma(z)=\dfrac{1}{1+e^{-z}}$")
    ax.axhline(.5, color="gray", ls=":", lw=1.2); ax.axvline(0, color="gray", ls=":", lw=1.2)
    ax.axhline(1, color="k", lw=.8, alpha=.35); ax.axhline(0, color="k", lw=.8, alpha=.35)
    ax.scatter([0], [.5], color=C4, s=80, zorder=6)
    ax.text(-7.7, .80, "vùng BÃO HOÀ\n$\\sigma\' \\approx 0$", color=C2,
            fontsize=8.8, va="top", bbox=BOX)
    ax.text(7.7, .30, "vùng BÃO HOÀ\n$\\sigma\' \\approx 0$", color=C2,
            fontsize=8.8, va="top", ha="right", bbox=BOX)
    ax.annotate("$\\sigma(0)=0.5$\nngưỡng quyết định mặc định", xy=(0, .5),
                xytext=(-7.7, .30), fontsize=8.8, color=C4, va="top",
                arrowprops=dict(arrowstyle="->", color=C4, lw=1.4), bbox=BOX)
    ax.set_ylim(-.14, 1.30)
    ax.set_xlabel("$z = w^Tx+b$  (logit)"); ax.set_ylabel("xác suất $\\hat{p}$")
    ax.legend(fontsize=10.5, loc="upper left", framealpha=.94)
    ax.set_title("Sigmoid ép mọi số thực về khoảng (0, 1)", fontsize=10.5)

    ax = axes[1]
    ax.plot(z, sg, color=C1, lw=1.6, alpha=.45, label=r"$\sigma(z)$")
    ax.plot(z, ds, color=C2, lw=2.8, label=r"$\sigma'(z)=\sigma(z)\,(1-\sigma(z))$")
    ax.axhline(.25, color=C4, ls="--", lw=1.4)
    ax.text(-7.7, .335, "đạo hàm cực đại = 0.25, đạt tại z = 0", color=C4,
            fontsize=8.8, va="center", bbox=BOX)
    ax.annotate("z càng xa 0 thì gradient càng tắt.\n"
                "Đây là gốc rễ của VANISHING GRADIENT\n(gặp lại ở Lab 09 - MLP).",
                xy=(5.0, float(ds[np.argmin(abs(z - 5.0))])),
                xytext=(1.35, .80), fontsize=8.8, color="#374151", va="top",
                arrowprops=dict(arrowstyle="->", color="dimgray", lw=1.4), bbox=BOX)
    ax.set_ylim(-.06, 1.22)
    ax.set_xlabel("z"); ax.set_ylabel("giá trị")
    ax.legend(fontsize=9.5, loc="upper left", framealpha=.94)
    ax.set_title("Đạo hàm của sigmoid", fontsize=10.5)

    fig.suptitle("Hàm sigmoid và đạo hàm của nó", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, .95])
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
    ax.text(.05, 9.4, "p = 0.5 → odds = 1  (hoà)\np = 0.9 → odds = 9  (9 ăn 1)", fontsize=8.7,
            color="#374151", va="top",
            bbox=dict(boxstyle="round,pad=0.32", facecolor="white", alpha=.93,
                      edgecolor="0.8"))

    ax = axes[1]
    ax.plot(p, np.log(odds), color=C3, lw=2.4)
    ax.axhline(0, color="gray", ls=":"); ax.axvline(.5, color="gray", ls=":")
    ax.set_xlabel("xác suất $p$"); ax.set_ylabel("logit $=\\ln\\frac{p}{1-p}$")
    ax.set_ylim(-7.5, 7.5)
    ax.set_title("Logit: $[0,1] \\rightarrow (-\\infty,\\infty)$", fontsize=10)
    ax.text(.50, -2.2, "Logit là hàm ngược của sigmoid.\n"
            "Logistic Regression thật ra là\nhồi quy TUYẾN TÍNH trên logit:\n"
            r"$\ln\frac{p}{1-p} = w^Tx+b$", fontsize=8.7, color="#374151", va="top",
            bbox=dict(boxstyle="round,pad=0.32", facecolor="white", alpha=.93,
                      edgecolor="0.8"))

    ax = axes[2]
    ws = np.array([0.2, 0.5, 1.0, 2.0])
    for w, col in zip(ws, [C1, C3, C4, C2]):
        ax.plot(np.linspace(-8, 8, 400), sigmoid(w * np.linspace(-8, 8, 400)),
                color=col, lw=2.1, label=f"$\\|w\\|$ = {w}  (OR = $e^w$ = {np.exp(w):.2f})")
    ax.axhline(.5, color="gray", ls=":")
    ax.set_xlabel("x"); ax.set_ylabel("$\\hat{p}$"); ax.legend(fontsize=8, loc="upper left", framealpha=.94)
    ax.set_ylim(-.06, 1.20)
    ax.set_title("Ảnh hưởng của $\\|w\\|$ lên độ dốc sigmoid", fontsize=10)
    fig.suptitle("Odds, logit và ý nghĩa của hệ số", fontweight="bold")
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
    ax.legend(fontsize=8.5); ax.set_title("Mất mát BCE theo xác suất dự đoán", fontsize=10)
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
    ax.set_title("BCE + sigmoid: hàm lồi", fontsize=10)
    ax.text(.03, .96, "chỉ một cực tiểu,\nGD luôn tìm được", transform=ax.transAxes,
            fontsize=8.3, va="top", color="#374151",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=.93,
                      edgecolor="0.8"))
    ax = axes[2]
    ax.plot(ws, mse, color=C2, lw=2.4)
    ax.set_xlabel("tham số w"); ax.set_ylabel("MSE")
    ax.set_title("MSE + sigmoid: không lồi", fontsize=10)
    ax.annotate("cao nguyên phẳng:\ngradient tắt, train đứng im", (-5, mse[20]),
                xytext=(-5.6, np.max(mse) * .62), fontsize=8.3,
                arrowprops=dict(arrowstyle="->", color="dimgray"), color="dimgray")
    fig.suptitle("Cross-Entropy so với MSE cho Logistic Regression",
                 fontweight="bold")
    fig.tight_layout()
    save(fig, "04_bce_vs_mse.png")


# ---------------------------------------------------------------- 5
def fig_decision_boundary():
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=260, n_features=2, n_redundant=0,
                               n_clusters_per_class=1, class_sep=1.05,
                               flip_y=0.02, random_state=4)
    clf = LogisticRegression().fit(X, y)
    pad = 1.0
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, 320),
                         np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, 320))
    P = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
    BOX = dict(boxstyle="round,pad=0.32", facecolor="white", alpha=.93, edgecolor="0.8")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.9))
    ax = axes[0]
    cf = ax.contourf(xx, yy, P, levels=np.linspace(0, 1, 21), cmap="RdBu_r", alpha=.7)
    ax.contour(xx, yy, P, levels=[.25, .75], colors="k", linewidths=.9, linestyles=":")
    ax.contour(xx, yy, P, levels=[.5], colors="k", linewidths=2.6)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu_r", s=24, edgecolor="k", linewidth=.45)

    # vector w vẽ ở góc trống, không đè lên dữ liệu
    w = clf.coef_[0]; wn = w / np.linalg.norm(w)
    x0 = np.array([xx.min() + 1.15, yy.max() - 1.25])
    ax.arrow(x0[0], x0[1], wn[0] * 1.15, wn[1] * 1.15, width=.055,
             color="#16a34a", ec="k", lw=.5, zorder=7, length_includes_head=True)
    ax.text(x0[0] - .35, x0[1] + .45,
            "$w$ vuông góc với ranh giới\nvà trỏ về phía lớp 1", fontsize=8.6,
            color="#166534", va="bottom", ha="left", zorder=8, bbox=BOX)
    ax.text(.03, .04, "ranh giới luôn là một đường thẳng\n"
                      "nét liền đậm: $\\hat{p}=0.5$ (ranh giới)\n"
                      "nét chấm: $\\hat{p}=0.25$ và $0.75$",
            transform=ax.transAxes, fontsize=8.4, va="bottom", bbox=BOX)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.set_title("Ranh giới quyết định $w^Tx+b=0$", fontsize=10)
    plt.colorbar(cf, ax=ax, label="$\\hat{p}(y=1\\,|\\,x)$")

    t = (X @ w + clf.intercept_[0]) / np.linalg.norm(w)
    ts = np.linspace(t.min() - .6, t.max() + .6, 400)
    ax = axes[1]
    ax.plot(ts, sigmoid(np.linalg.norm(w) * ts), color=C1, lw=2.8, zorder=3)
    jitter = np.random.default_rng(0).uniform(-.035, .035, len(t))
    ax.scatter(t, y + jitter, c=y, cmap="RdBu_r", s=26, edgecolor="k",
               linewidth=.4, zorder=4, alpha=.85)
    ax.axhline(.5, color="gray", ls=":", lw=1.2)
    ax.axvline(0, color="k", ls="--", lw=1.8)
    ax.text(1.35, .34, "ranh giới\nquyết định", fontsize=8.6, va="center", bbox=BOX)
    ax.set_ylim(-.22, 1.22)
    ax.set_xlabel("khoảng cách CÓ DẤU tới ranh giới")
    ax.set_ylabel("$\\hat{p}$")
    ax.text(.03, .96, "bài toán thu về đúng\nmột đường sigmoid 1 chiều",
            transform=ax.transAxes, fontsize=8.5, va="top", bbox=BOX)
    ax.set_title("Chiếu dữ liệu lên hướng $w$", fontsize=10)
    fig.suptitle("Hình học của Logistic Regression", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, .95])
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
    BOXW = dict(boxstyle="round,pad=0.3", facecolor="white", alpha=.92, edgecolor="0.8")
    axes[0].text(-3.9, 2.35, "phạt MẠNH:\nchuyển tiếp rất mờ", fontsize=8.3, color="k", va="top", bbox=BOXW)
    axes[2].text(-3.9, 2.35, "phạt YẾU:\nchuyển tiếp gắt như bậc thang", fontsize=8.3, color="k", va="top", bbox=BOXW)

    Cgrid = np.geomspace(1e-4, 1e10, 36)
    # tol chặt: để mặc định (1e-4) thì solver dừng sớm và đường cong PHẲNG giả tạo
    norms = [np.linalg.norm(
        LogisticRegression(C=c, max_iter=200000, tol=1e-12).fit(Xa, ya).coef_)
        for c in Cgrid]
    ax = axes[3]
    ax.plot(Cgrid, norms, "o-", color=C2, ms=3.8)
    ax.set_xscale("log")          # trục y TUYẾN TÍNH: đường thẳng đi lên = tăng theo log(C)
    ax.set_xlabel("C  (= $1/\\alpha$, càng lớn càng ÍT phạt)")
    ax.set_ylabel("$\\|w\\|$")
    ax.set_title("Chuẩn $\\|w\\|$ theo tham số C", fontsize=10)
    ax.text(.03, .97,
            "dữ liệu ở đây tách hoàn toàn nên\nnghiệm MLE nằm ở vô cực: trục x\n"
            "là log, trục y tuyến tính, nên\nđường thẳng nghĩa là $\\|w\\|$ vẫn\n"
            "tăng mãi theo $\\log C$ khi bỏ phạt",
            transform=ax.transAxes, fontsize=7.6, color="#374151", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=.92,
                      edgecolor="0.8"))
    fig.suptitle("Ảnh hưởng của tham số C lên hệ số", fontweight="bold")
    fig.tight_layout()
    save(fig, "06_regularization_C.png")


# ---------------------------------------------------------------- 7
def fig_polynomial_logistic():
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    X, y = make_moons(n_samples=400, noise=.30, random_state=42)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.5, random_state=0, stratify=y)
    xx, yy = np.meshgrid(np.linspace(-2.2, 3.2, 300), np.linspace(-1.8, 2.2, 300))
    BOX = dict(boxstyle="round,pad=0.3", facecolor="white", alpha=.93, edgecolor="0.8")

    fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.2))
    for ax, d in zip(axes, [1, 3, 6, 20]):
        m = make_pipeline(PolynomialFeatures(d), StandardScaler(),
                          LogisticRegression(C=1e6, max_iter=20000)).fit(Xtr, ytr)
        P = m.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
        ax.contourf(xx, yy, P, levels=np.linspace(0, 1, 21), cmap="RdBu_r", alpha=.72)
        ax.contour(xx, yy, P, levels=[.5], colors="k", linewidths=2.2)
        ax.scatter(Xtr[:, 0], Xtr[:, 1], c=ytr, cmap="RdBu_r", s=20,
                   edgecolor="k", linewidth=.35)
        tr_a, te_a = m.score(Xtr, ytr) * 100, m.score(Xte, yte) * 100
        note = {1: "thẳng, underfit", 3: "cong vừa", 6: "bám sát hai vành trăng",
                20: "uốn éo, overfit"}[d]
        ax.set_title(f"degree = {d}  ({note})", fontsize=10)
        ax.text(.03, .04, f"train acc = {tr_a:.1f}%\ntest  acc = {te_a:.1f}%",
                transform=ax.transAxes, fontsize=8.8, va="bottom",
                fontweight="bold" if d == 20 else "normal", bbox=BOX)
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    fig.suptitle("Ranh giới cong nhờ polynomial features", fontweight="bold", y=.995)
    fig.text(.5, .925, "các chấm hiển thị là tập train", ha="center", va="top",
             fontsize=9, color="#374151")
    fig.tight_layout(rect=[0, 0, 1, .878])
    save(fig, "07_polynomial_logistic.png")


# ---------------------------------------------------------------- 8
def fig_softmax():
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=300, centers=3, cluster_std=1.15, random_state=7)
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1.5, X[:, 0].max() + 1.5, 320),
                         np.linspace(X[:, 1].min() - 1.5, X[:, 1].max() + 1.5, 320))
    grid = np.c_[xx.ravel(), yy.ravel()]

    PT_COLORS = ["#1d4ed8", "#15803d", "#b91c1c"]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    z = np.linspace(-3, 3, 400)
    ax = axes[0]
    logits = np.vstack([1.4 * z, .2 * z + .6, -1.1 * z + .2])
    e = np.exp(logits - logits.max(0)); sm = e / e.sum(0)
    for i, col in enumerate(PT_COLORS):
        ax.plot(z, sm[i], color=col, lw=2.3, label=f"$\\hat{{p}}$(lớp {i})")
    ax.set_xlabel("một lát cắt của không gian đặc trưng"); ax.set_ylabel("xác suất")
    ax.legend(fontsize=8.5)
    ax.set_title("Softmax: ba xác suất cộng lại bằng 1\n"
                 r"$\hat{p}_c=e^{z_c}/\sum_k e^{z_k}$", fontsize=10)

    from sklearn.multiclass import OneVsRestClassifier
    models = [
        (LogisticRegression(max_iter=2000), "multinomial (softmax thật)",
         "một bài toán tối ưu chung cho cả 3 lớp"),
        (OneVsRestClassifier(LogisticRegression(max_iter=2000)), "one-vs-rest",
         "3 bộ phân loại nhị phân riêng rồi so điểm"),
    ]
    for ax, (est, ttl, note) in zip(axes[1:], models):
        clf = est.fit(X, y)
        Z = clf.predict(grid).reshape(xx.shape)
        ax.contourf(xx, yy, Z, levels=[-.5, .5, 1.5, 2.5],
                    colors=["#bfdbfe", "#bbf7d0", "#fecaca"])
        ax.contour(xx, yy, Z, levels=[.5, 1.5], colors="k", linewidths=1.8)
        # màu điểm PHẢI khớp màu vùng: lớp 0 xanh dương, lớp 1 xanh lá, lớp 2 đỏ
        for c, col in zip([0, 1, 2], PT_COLORS):
            ax.scatter(X[y == c, 0], X[y == c, 1], color=col, s=22,
                       edgecolor="k", linewidth=.35, label=f"lớp {c}")
        ax.legend(fontsize=7.8, loc="upper right", framealpha=.94)
        ax.set_title(f"{ttl}   acc = {clf.score(X, y)*100:.1f}%", fontsize=9.5)
        ax.text(.03, .03, note, transform=ax.transAxes, fontsize=8.2, va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=.93,
                          edgecolor="0.8"))
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
    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax * 1.26)
    for t, col in [(.3, C3), (.5, "k"), (.7, C4)]:
        ax.axvline(t, color=col, ls="--", lw=1.9)
        ax.text(t, ymax * 1.13, f" {t}", color=col, fontsize=9, fontweight="bold")
    ax.set_xlabel("$\\hat{p}$ model xuất ra"); ax.set_ylabel("số mẫu")
    ax.legend(fontsize=8.5, loc="center right", framealpha=.94)
    ax.set_title("Phân phối điểm số và ba ngưỡng", fontsize=10)

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
    ax.annotate(f"F1 đạt cực đại tại ngưỡng {tb:.2f},\nkhông phải tại 0.5",
                xy=(tb, max(F)), xytext=(.10, 1.30), fontsize=8.5, color=C3,
                fontweight="bold", va="top",
                arrowprops=dict(arrowstyle="->", color=C3, lw=1.4),
                bbox=dict(boxstyle="round,pad=0.32", facecolor="white",
                          alpha=.93, edgecolor="0.8"))
    ax.set_xlabel("ngưỡng"); ax.set_ylabel("giá trị")
    ax.set_ylim(-.04, 1.34)
    ax.legend(fontsize=8.5, loc="lower left", framealpha=.94)
    ax.set_title("Precision, Recall, F1 theo ngưỡng", fontsize=10)

    fpr, tpr, thr = roc_curve(yte, pr)
    ax = axes[2]
    ax.plot(fpr, tpr, color=C1, lw=2.4, label=f"ROC (AUC = {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="đoán mò")
    for t, col in [(.3, C3), (.5, "k"), (.7, C4)]:
        i = int(np.argmin(np.abs(thr - t)))
        ax.scatter([fpr[i]], [tpr[i]], color=col, s=75, zorder=5)
        ax.text(fpr[i] + .03, tpr[i] - .06, f"ngưỡng {t}", color=col, fontsize=8.3)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR (Recall)"); ax.legend(fontsize=8.5, loc="lower right")
    ax.set_title("Đường ROC và vị trí của từng ngưỡng", fontsize=10)
    fig.suptitle("Ngưỡng quyết định và các chỉ số đánh giá", fontweight="bold",
                 y=.995)
    fig.text(.5, .925, "đổi ngưỡng không phải train lại model, chỉ là đọc xác suất "
             "theo cách khác", ha="center", va="top", fontsize=9, color="#374151")
    fig.tight_layout(rect=[0, 0, 1, .875])
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
