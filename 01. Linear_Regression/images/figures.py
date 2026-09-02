"""
Sinh toàn bộ hình minh hoạ cho Lab 01 - Linear Regression.

Chạy:  python figures.py
Kết quả: các file .png trong cùng thư mục, được nhúng vào notebook bằng markdown.
Sinh viên có thể sửa script này để tự thí nghiệm với tham số.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon

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


# ---------------------------------------------------------------- 1
def fig_regression_vs_classification():
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))

    x = np.linspace(20, 200, 60)
    y = 50 + 0.3 * x + rng.normal(0, 6, 60)
    ax[0].scatter(x, y, s=22, color=C1, alpha=.75)
    ax[0].plot(x, 50 + .3 * x, color=C2, lw=2)
    ax[0].set_title("HỒI QUY: output là số thực liên tục")
    ax[0].set_xlabel("Diện tích (m²)"); ax[0].set_ylabel("Giá (triệu)")
    ax[0].annotate("dự đoán = một con số\n(101.4 triệu)", xy=(160, 98), xytext=(60, 115),
                   arrowprops=dict(arrowstyle="->", color="gray"), fontsize=9, color="dimgray")

    xa = rng.normal(2.2, .55, 40); ya = rng.normal(2.2, .55, 40)
    xb = rng.normal(4.2, .55, 40); yb = rng.normal(4.0, .55, 40)
    ax[1].scatter(xa, ya, s=22, color=C1, alpha=.8, label="lớp 0")
    ax[1].scatter(xb, yb, s=22, color=C2, alpha=.8, marker="s", label="lớp 1")
    xx = np.linspace(.6, 5.8, 10)
    ax[1].plot(xx, 6.6 - xx, "k--", lw=1.8)
    ax[1].set_title("PHÂN LOẠI: output là nhãn rời rạc")
    ax[1].set_xlabel("feature 1"); ax[1].set_ylabel("feature 2"); ax[1].legend(fontsize=8)
    fig.suptitle("Hai họ bài toán học có giám sát", fontweight="bold")
    save(fig, "01_hoiquy_vs_phanloai.png")


# ---------------------------------------------------------------- 2
def fig_residuals_geometry():
    rng = np.random.default_rng(3)
    x = np.linspace(1, 9, 12)
    y = 2.1 * x + 3 + rng.normal(0, 2.4, 12)
    w, b = np.polyfit(x, y, 1)
    yh = w * x + b

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.scatter(x, y, s=48, color=C1, zorder=3, label="dữ liệu thật $y_i$")
    ax.plot(x, yh, color=C2, lw=2.2, label=r"đường dự đoán $\hat{y}=wx+b$")
    for xi, yi, yhi in zip(x, y, yh):
        ax.plot([xi, xi], [yi, yhi], color="gray", lw=1.4, ls="--", zorder=2)
        side = abs(yi - yhi)
        ax.add_patch(plt.Rectangle((xi, min(yi, yhi)), side * .55, side,
                                   color=C4, alpha=.28, zorder=1))
    ax.plot([], [], color="gray", ls="--", label=r"sai số $e_i=\hat{y}_i-y_i$")
    ax.add_patch(plt.Rectangle((0, 0), 0, 0, color=C4, alpha=.4, label=r"$e_i^2$ (MSE cộng các ô này)"))
    ax.set_title("MSE = trung bình DIỆN TÍCH các hình vuông sai số\n"
                 "→ điểm càng xa đường, phạt càng nặng (bình phương)", fontsize=10)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.legend(fontsize=8, loc="upper left")
    save(fig, "02_mse_hinh_vuong_sai_so.png")


# ---------------------------------------------------------------- 3
def fig_loss_surface_gd():
    rng = np.random.default_rng(42)
    N = 60
    x = np.linspace(0, 10, N)
    y = 3 * x + 5 + rng.normal(0, 1.5, N)

    def mse(w, b):
        return np.mean((w * x[:, None, None] + b - y[:, None, None]) ** 2, axis=0)

    W, B = np.meshgrid(np.linspace(-1, 7, 160), np.linspace(-6, 16, 160))
    Z = mse(W, B)

    # chạy GD lưu đường đi
    w, b, lr = -0.5, 14.0, 0.012
    path = [(w, b)]
    for _ in range(60):
        e = w * x + b - y
        w -= lr * 2 * (e * x).mean()
        b -= lr * 2 * e.mean()
        path.append((w, b))
    path = np.array(path)

    fig = plt.figure(figsize=(11.5, 4.4))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(W, B, Z, cmap="viridis", alpha=.85, linewidth=0, antialiased=True)
    ax1.set_xlabel("w"); ax1.set_ylabel("b"); ax1.set_zlabel("MSE")
    ax1.set_title("Mặt mất mát MSE(w, b) — một cái bát lồi")
    ax1.view_init(elev=34, azim=-128)

    ax2 = fig.add_subplot(1, 2, 2)
    cs = ax2.contour(W, B, Z, levels=np.geomspace(Z.min() + .5, Z.max(), 18), cmap="viridis")
    ax2.clabel(cs, inline=True, fontsize=6, fmt="%.0f")
    ax2.plot(path[:, 0], path[:, 1], "o-", color=C2, ms=3.4, lw=1.5, label="đường đi của Gradient Descent")
    ax2.scatter([3], [5], marker="*", s=280, color=C4, zorder=5, edgecolor="k",
                linewidth=.6, label="nghiệm tối ưu (3, 5)")
    ax2.scatter([path[0, 0]], [path[0, 1]], s=60, color="k", zorder=5, label="điểm khởi tạo")
    ax2.set_xlabel("w"); ax2.set_ylabel("b"); ax2.legend(fontsize=8)
    ax2.set_title("Nhìn từ trên xuống: GD đi vuông góc với đường đồng mức")
    fig.suptitle("MSE của hồi quy tuyến tính là hàm LỒI → chỉ có 1 cực tiểu toàn cục",
                 fontweight="bold")
    save(fig, "03_mat_mat_mat_va_duong_di_gd.png")


# ---------------------------------------------------------------- 4
def fig_learning_rate():
    rng = np.random.default_rng(42)
    N = 60
    x = np.linspace(0, 10, N)
    y = 3 * x + 5 + rng.normal(0, 1.5, N)

    def run(lr, steps=40):
        w, b = -0.5, 14.0
        P, L = [(w, b)], []
        for _ in range(steps):
            e = w * x + b - y
            L.append((e ** 2).mean())
            gw, gb = 2 * (e * x).mean(), 2 * e.mean()
            w -= lr * gw; b -= lr * gb
            if not np.isfinite(w) or abs(w) > 1e4:
                break
            P.append((w, b))
        return np.array(P), np.array(L)

    W, B = np.meshgrid(np.linspace(-2, 8, 140), np.linspace(-8, 18, 140))
    Z = np.mean((W[None] * x[:, None, None] + B[None] - y[:, None, None]) ** 2, axis=0)

    cfg = [(0.0008, "lr QUÁ NHỎ (0.0008)\n→ bò chậm, 40 bước chưa tới nơi", C1),
           (0.012, "lr VỪA (0.012)\n→ hội tụ gọn gàng", C3),
           (0.0255, "lr QUÁ LỚN (0.0255)\n→ nảy qua nảy lại / phân kỳ", C2)]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.6),
                             gridspec_kw={"height_ratios": [1.35, 1]})
    for j, (lr, title, col) in enumerate(cfg):
        P, L = run(lr)
        a = axes[0, j]
        a.contour(W, B, Z, levels=np.geomspace(Z.min() + .5, Z.max(), 16),
                  cmap="Greys", linewidths=.7)
        a.plot(P[:, 0], P[:, 1], "o-", color=col, ms=3, lw=1.4)
        a.scatter([3], [5], marker="*", s=200, color=C4, edgecolor="k", zorder=5)
        a.set_title(title, fontsize=9.5); a.set_xlabel("w"); a.set_ylabel("b")
        b_ = axes[1, j]
        b_.plot(L, color=col, lw=1.8)
        b_.set_yscale("log"); b_.set_xlabel("bước"); b_.set_ylabel("MSE (log)")
    fig.suptitle("Learning rate quyết định số phận của Gradient Descent", fontweight="bold")
    fig.tight_layout()
    save(fig, "04_anh_huong_learning_rate.png")


# ---------------------------------------------------------------- 5
def fig_normal_equation_projection():
    fig = plt.figure(figsize=(7.4, 5.6))
    ax = fig.add_subplot(111, projection="3d")

    # mặt phẳng = không gian cột của X (mọi tổ hợp tuyến tính của 2 cột)
    P = np.array([1.0, .22, 0.]); Q = np.array([.18, 1.0, 0.])
    g = np.linspace(-1.9, 1.9, 12)
    G1, G2 = np.meshgrid(g, g)
    S = G1[..., None] * P + G2[..., None] * Q
    ax.plot_surface(S[..., 0], S[..., 1], S[..., 2], alpha=.22, color=C1,
                    edgecolor=C1, linewidth=.25, rstride=1, cstride=1)

    yhat = np.array([1.05, .78, 0.])
    y = yhat + np.array([0., 0., 1.35])

    ax.quiver(0, 0, 0, *y, color=C2, lw=3.0, arrow_length_ratio=.10)
    ax.quiver(0, 0, 0, *yhat, color=C3, lw=3.0, arrow_length_ratio=.13)
    ax.plot(*zip(yhat, y), color="dimgray", ls="--", lw=2.0)
    ax.quiver(0, 0, 0, *P, color="k", lw=1.6, arrow_length_ratio=.16)
    ax.quiver(0, 0, 0, *Q, color="k", lw=1.6, arrow_length_ratio=.16)

    # ký hiệu góc vuông tại chân đường vuông góc
    d = .16
    corner = [yhat + np.array([-d * .9, -d * .2, 0]),
              yhat + np.array([-d * .9, -d * .2, d]),
              yhat + np.array([0, 0, d])]
    ax.plot(*zip(*corner), color="dimgray", lw=1.3)

    ax.text(y[0] + .06, y[1], y[2] + .06, "$y$  (giá trị thật)", color=C2,
            fontsize=11, fontweight="bold")
    ax.text(yhat[0] + .16, yhat[1] + .42, -.30, "$\\hat{y}=X\\theta$\n(dự đoán)",
            color=C3, fontsize=10.5, fontweight="bold")
    ax.text(y[0] - .30, y[1] + 1.15, y[2] - .78,
            "phần dư $e=y-\\hat{y}$\n$\\perp$ mặt phẳng", color="dimgray", fontsize=9.5)
    ax.text(P[0] - .05, P[1] - .78, .05, "cột $x_1$", fontsize=9.5)
    ax.text(Q[0] - .62, Q[1] + .12, .04, "cột $x_2$", fontsize=9.5)
    ax.text(-2.4, -1.5, -.06, "không gian cột của $X$\n(mọi tổ hợp tuyến tính $X\\theta$)",
            color=C1, fontsize=9.5)

    ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2); ax.set_zlim(-.35, 1.6)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_box_aspect((1, 1, .62))
    ax.set_title("Ý nghĩa hình học của Normal Equation\n"
                 r"$\hat{y}$ là HÌNH CHIẾU VUÔNG GÓC của $y$ xuống không gian cột của $X$"
                 "\n" r"$X^T(y-X\theta)=0 \;\Rightarrow\; \hat{\theta}=(X^TX)^{-1}X^Ty$",
                 fontsize=10.5, pad=2)
    ax.view_init(elev=17, azim=-62)
    save(fig, "05_normal_equation_hinh_chieu.png")


# ---------------------------------------------------------------- 6
def fig_residual_diagnostics():
    rng = np.random.default_rng(7)
    n = 120
    x = np.linspace(0, 10, n)
    cases = [
        ("ĐẠT: phần dư ngẫu nhiên quanh 0", 2 * x + 3 + rng.normal(0, 1.2, n), C3),
        ("LỖI: quan hệ phi tuyến còn sót\n→ thêm $x^2$ hoặc đổi model", 0.55 * (x - 5) ** 2 + rng.normal(0, 1.0, n), C4),
        ("LỖI: phương sai tăng dần (heteroscedasticity)\n→ log-transform y hoặc dùng WLS", 2 * x + 3 + rng.normal(0, .25 + .45 * x, n), C2),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
    for ax, (title, y, col) in zip(axes, cases):
        w, b = np.polyfit(x, y, 1)
        r = y - (w * x + b)
        ax.axhline(0, color="k", lw=1.2)
        ax.scatter(w * x + b, r, s=18, color=col, alpha=.75)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("giá trị dự đoán $\\hat{y}$"); ax.set_ylabel("phần dư $y-\\hat{y}$")
    fig.suptitle("Residual plot — công cụ chẩn đoán số 1 của hồi quy tuyến tính",
                 fontweight="bold")
    fig.tight_layout()
    save(fig, "06_chan_doan_phan_du.png")


# ---------------------------------------------------------------- 7
def fig_bias_variance():
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import mean_squared_error

    rng = np.random.default_rng(1)
    xt = np.sort(rng.uniform(0, 2 * np.pi, 30)); yt = np.sin(xt) + rng.normal(0, .22, 30)
    xv = np.sort(rng.uniform(0, 2 * np.pi, 200)); yv = np.sin(xv) + rng.normal(0, .22, 200)
    xx = np.linspace(0, 2 * np.pi, 400).reshape(-1, 1)

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.6))
    for ax, (d, lab, col) in zip(axes[:3], [(1, "degree 1 — UNDERFIT\n(bias cao)", C1),
                                            (4, "degree 4 — VỪA ĐẸP", C3),
                                            (15, "degree 15 — OVERFIT\n(variance cao)", C2)]):
        m = make_pipeline(PolynomialFeatures(d), LinearRegression()).fit(xt.reshape(-1, 1), yt)
        ax.scatter(xt, yt, s=26, color="k", alpha=.65, label="train (30 điểm)")
        ax.plot(xx, np.sin(xx), "--", color="gray", lw=1.6, label="hàm thật sin(x)")
        ax.plot(xx, m.predict(xx), color=col, lw=2.2, label="model")
        ax.set_ylim(-2.1, 2.1); ax.set_title(lab, fontsize=9.5); ax.legend(fontsize=7)

    degs = range(1, 16)
    tr, te = [], []
    for d in degs:
        m = make_pipeline(PolynomialFeatures(d), LinearRegression()).fit(xt.reshape(-1, 1), yt)
        tr.append(np.sqrt(mean_squared_error(yt, m.predict(xt.reshape(-1, 1)))))
        te.append(np.sqrt(mean_squared_error(yv, m.predict(xv.reshape(-1, 1)))))
    ax = axes[3]
    ax.plot(list(degs), tr, "o-", color=C1, label="train RMSE")
    ax.plot(list(degs), te, "s-", color=C2, label="test RMSE")
    ax.axvline(int(np.argmin(te)) + 1, color=C3, ls="--", lw=1.6)
    ax.text(int(np.argmin(te)) + 1.25, max(te) * .55, "điểm ngọt", color=C3, fontsize=9)
    ax.set_yscale("log"); ax.set_xlabel("bậc đa thức (độ phức tạp)"); ax.set_ylabel("RMSE (log)")
    ax.set_title("Đường cong bias–variance", fontsize=9.5); ax.legend(fontsize=8)
    fig.suptitle("Train RMSE giảm mãi, Test RMSE giảm rồi TĂNG — đó là overfitting",
                 fontweight="bold")
    fig.tight_layout()
    save(fig, "07_bias_variance_tradeoff.png")


# ---------------------------------------------------------------- 8
def fig_ridge_lasso():
    from sklearn.linear_model import Lasso

    # Dạng toàn phương thật của MSE: Q(w) = (w-c)^T A (w-c)
    A = np.array([[0.80, 0.80], [0.80, 1.00]])
    c = np.array([2.0, 1.55])

    def Q(w1, w2):
        d1, d2 = w1 - c[0], w2 - c[1]
        return A[0, 0] * d1 ** 2 + 2 * A[0, 1] * d1 * d2 + A[1, 1] * d2 ** 2

    G1, G2 = np.meshgrid(np.linspace(-2.2, 4.0, 400), np.linspace(-2.0, 3.4, 400))
    ZQ = Q(G1, G2)

    fig = plt.figure(figsize=(14, 4.6))
    specs = [
        ("Ridge (L2):  $w_1^2+w_2^2 \\leq t$", "l2",
         "Vùng ràng buộc TRÒN, không có góc.\nĐiểm tiếp xúc hầu như luôn có\ncả hai toạ độ khác 0\n→ hệ số bị CO NHỎ, không về 0."),
        ("Lasso (L1):  $|w_1|+|w_2| \\leq t$", "l1",
         "Vùng ràng buộc có GÓC NHỌN nằm\nngay trên trục toạ độ. Ellipse rất dễ\nchạm đúng vào góc\n→ hệ số bằng ĐÚNG 0."),
    ]
    for k, (name, kind, txt) in enumerate(specs):
        ax = fig.add_subplot(1, 3, k + 1)

        # tìm nghiệm bị ràng buộc bằng cách quét biên của vùng khả thi
        th = np.linspace(0, 2 * np.pi, 4000)
        if kind == "l2":
            bx, by = np.cos(th), np.sin(th)
            ax.add_patch(Circle((0, 0), 1, color=C1, alpha=.28, zorder=1))
        else:
            u = np.linspace(-1, 1, 2000)
            bx = np.r_[u, u[::-1]]
            by = np.r_[1 - np.abs(u), -(1 - np.abs(u[::-1]))]
            ax.add_patch(Polygon([(1, 0), (0, 1), (-1, 0), (0, -1)],
                                 color=C1, alpha=.28, zorder=1))
        vals = Q(bx, by)
        i = int(np.argmin(vals))
        hit, lvl = (bx[i], by[i]), vals[i]

        # các đường đồng mức MSE, tô đậm đúng đường tiếp xúc
        ax.contour(G1, G2, ZQ, levels=np.sort(np.r_[lvl * np.array([.12, .38, .68]), lvl,
                                                    lvl * np.array([1.45, 2.1])]),
                   colors=C2, linewidths=[.9, .9, .9, 2.2, .9, .9], zorder=2)
        ax.scatter(*c, color=C2, s=55, zorder=6)
        ax.text(c[0] + .14, c[1] + .1, "nghiệm OLS\n(không phạt)", color=C2, fontsize=8.5)
        ax.scatter(*hit, color="k", s=85, zorder=7)
        ax.annotate("nghiệm sau khi phạt\n$w$ = ({:.2f}, {:.2f})".format(*hit), hit,
                    xytext=(-2.05, 2.75), fontsize=8.5,
                    arrowprops=dict(arrowstyle="->", color="k", lw=1.2))
        ax.axhline(0, color="k", lw=.9); ax.axvline(0, color="k", lw=.9)
        ax.set_xlim(-2.2, 4.0); ax.set_ylim(-2.0, 3.4)
        ax.set_xlabel("$w_1$"); ax.set_ylabel("$w_2$")
        ax.set_title(name, fontsize=10.5)
        ax.text(-2.1, -1.92, txt, fontsize=7.8, color="dimgray", va="bottom")
        ax.set_aspect("equal")

    # (c) đường đi của hệ số Lasso
    rng = np.random.default_rng(0)
    n, p_ = 80, 8
    X = rng.normal(size=(n, p_)); X[:, 1] = X[:, 0] * .95 + rng.normal(0, .2, n)
    beta = np.array([3., 0., -2., 0., 1.5, 0., 0., 0.])
    y = X @ beta + rng.normal(0, .8, n)
    alphas = np.geomspace(1e-3, 30, 60)
    coefs = np.array([Lasso(alpha=a, max_iter=20000).fit(X, y).coef_ for a in alphas])
    ax = fig.add_subplot(1, 3, 3)
    for i in range(p_):
        real = beta[i] != 0
        ax.plot(alphas, coefs[:, i], lw=2.0 if real else 1.1,
                alpha=1.0 if real else .55,
                label=("$w_%d$ (thật)" % i) if real else ("$w_%d$ (nhiễu)" % i if i == 1 else None))
    ax.set_xscale("log"); ax.axhline(0, color="k", lw=1)
    ax.set_xlabel(r"$\alpha$ (mức phạt) — tăng dần $\rightarrow$")
    ax.set_ylabel("giá trị hệ số")
    ax.set_title("Lasso path: hệ số lần lượt bị ép về ĐÚNG 0\n→ tự động chọn feature", fontsize=10.5)
    ax.legend(fontsize=7.5, ncol=2, loc="lower left")
    fig.suptitle("Vì sao Lasso chọn được feature còn Ridge thì không", fontweight="bold")
    fig.tight_layout()
    save(fig, "08_ridge_vs_lasso.png")


# ---------------------------------------------------------------- 9
def fig_outlier_robustness():
    from sklearn.linear_model import LinearRegression, HuberRegressor
    rng = np.random.default_rng(5)
    x = np.linspace(0, 10, 60)
    y = 3 * x + 5 + rng.normal(0, 1.5, 60)
    xo = np.r_[x, [2., 3., 4.]]; yo = np.r_[y, [70., 75., 72.]]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    for ax, (xa, ya, ttl) in zip(axes[:2], [(x, y, "Không có outlier"),
                                            (xo, yo, "Thêm 3 outlier ở y≈72")]):
        ols = LinearRegression().fit(xa.reshape(-1, 1), ya)
        hub = HuberRegressor().fit(xa.reshape(-1, 1), ya)
        ax.scatter(xa, ya, s=22, color=C1, alpha=.7)
        xs = np.linspace(0, 10, 10).reshape(-1, 1)
        ax.plot(xs, 3 * xs + 5, "--", color="gray", lw=1.8, label="true: w=3, b=5")
        ax.plot(xs, ols.predict(xs), color=C2, lw=2,
                label=f"OLS/MSE: w={ols.coef_[0]:.2f}")
        ax.plot(xs, hub.predict(xs), color=C3, lw=2,
                label=f"Huber: w={hub.coef_[0]:.2f}")
        ax.set_title(ttl, fontsize=10); ax.legend(fontsize=8); ax.set_xlabel("x"); ax.set_ylabel("y")

    e = np.linspace(-4, 4, 400)
    delta = 1.35
    hub_l = np.where(np.abs(e) <= delta, .5 * e ** 2, delta * (np.abs(e) - .5 * delta))
    ax = axes[2]
    ax.plot(e, e ** 2, color=C2, lw=2, label="MSE: $e^2$ (phạt bùng nổ)")
    ax.plot(e, np.abs(e), color=C1, lw=2, label="MAE: $|e|$")
    ax.plot(e, hub_l, color=C3, lw=2.4, ls="--", label=r"Huber ($\delta$=1.35)")
    ax.set_xlabel("sai số e"); ax.set_ylabel("mất mát"); ax.legend(fontsize=8)
    ax.set_title("Ba hàm mất mát: MSE phạt outlier nặng nhất", fontsize=10)
    fig.suptitle("Một vài outlier đủ kéo lệch cả đường hồi quy khi dùng MSE", fontweight="bold")
    fig.tight_layout()
    save(fig, "09_outlier_mse_vs_huber.png")


if __name__ == "__main__":
    fig_regression_vs_classification()
    fig_residuals_geometry()
    fig_loss_surface_gd()
    fig_learning_rate()
    fig_normal_equation_projection()
    fig_residual_diagnostics()
    fig_bias_variance()
    fig_ridge_lasso()
    fig_outlier_robustness()
    print("Xong.")
