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
    ax[0].set_ylim(y.min() - 6, y.max() + 14)
    ax[0].annotate("mỗi diện tích cho ra\nMỘT con số (98.0 triệu)", xy=(160, 50 + .3 * 160),
                   xytext=(24, y.max() + 3), fontsize=8.8, color="#374151",
                   arrowprops=dict(arrowstyle="->", color="gray", lw=1.3),
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                             alpha=.9, edgecolor="0.8"))

    xa = rng.normal(2.2, .55, 40); ya = rng.normal(2.2, .55, 40)
    xb = rng.normal(4.2, .55, 40); yb = rng.normal(4.0, .55, 40)
    ax[1].scatter(xa, ya, s=22, color=C1, alpha=.8, label="lớp 0")
    ax[1].scatter(xb, yb, s=22, color=C2, alpha=.8, marker="s", label="lớp 1")
    ax[1].set_ylim(.3, 7.0)
    xx = np.linspace(.6, 5.8, 10)
    ax[1].plot(xx, 6.6 - xx, "k--", lw=1.8)
    ax[1].set_title("PHÂN LOẠI: output là nhãn rời rạc")
    ax[1].set_xlabel("feature 1"); ax[1].set_ylabel("feature 2")
    ax[1].legend(fontsize=8, loc="lower left", framealpha=.92)
    fig.suptitle("Hai họ bài toán học có giám sát", fontweight="bold")
    save(fig, "01_hoiquy_vs_phanloai.png")


# ---------------------------------------------------------------- 2
def fig_residuals_geometry():
    rng = np.random.default_rng(3)
    x = np.linspace(1, 11, 11)
    y = 0.62 * x + 4 + rng.normal(0, 1.05, 11)
    w, b = np.polyfit(x, y, 1)
    yh = w * x + b

    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    for xi, yi, yhi in zip(x, y, yh):
        e = yi - yhi
        # cạnh hình vuông = |e| trên CẢ hai trục (nhờ set_aspect("equal") ở dưới)
        ax.add_patch(plt.Rectangle((xi, min(yi, yhi)), abs(e), abs(e),
                                   facecolor=C4, alpha=.30, edgecolor=C4, lw=1.1, zorder=1))
        ax.plot([xi, xi], [yi, yhi], color="dimgray", lw=1.5, ls="--", zorder=2)
    ax.plot(np.r_[x.min() - .4, x.max() + .4],
            w * np.r_[x.min() - .4, x.max() + .4] + b, color=C2, lw=2.4, zorder=3,
            label=r"đường dự đoán $\hat{y}=wx+b$")
    ax.scatter(x, y, s=52, color=C1, zorder=4, edgecolor="white", linewidth=.8,
               label=r"dữ liệu thật $y_i$")
    ax.plot([], [], color="dimgray", ls="--", label=r"sai số $e_i=y_i-\hat{y}_i$")
    ax.add_patch(plt.Rectangle((0, 0), 0, 0, facecolor=C4, alpha=.5, edgecolor=C4,
                               label=r"hình vuông cạnh $|e_i|$, diện tích $e_i^2$"))

    ax.set_aspect("equal")          # BẮT BUỘC: có vậy hình vuông mới thật sự vuông
    ax.set_xlim(x.min() - .8, x.max() + 3.6)
    ax.set_ylim(y.min() - 1.4, y.max() + 2.6)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(fontsize=8.4, loc="upper left", framealpha=.95)
    ax.set_title("MSE = trung bình DIỆN TÍCH các hình vuông sai số\n"
                 "điểm xa gấp đôi → ô vuông rộng gấp 4 → bị phạt gấp 4",
                 fontsize=10.5)
    save(fig, "02_mse_hinh_vuong_sai_so.png")


# ---------------------------------------------------------------- 3
def fig_loss_surface_gd():
    """Feature KHÔNG chuẩn hoá -> Hessian có số điều kiện ~137 -> contour dẹt như khe hẹp.
    GD lao rất nhanh vào khe rồi BÒ rất chậm dọc theo khe. Đây chính là lý do
    mục 4.2c nói phải scale feature trước khi chạy Gradient Descent."""
    rng = np.random.default_rng(42)
    N = 60
    x = np.linspace(0, 10, N)
    y = 3 * x + 5 + rng.normal(0, 1.5, N)

    X = np.column_stack([np.ones(N), x])
    ev = np.linalg.eigvalsh(2 * (X.T @ X) / N)
    kappa = ev.max() / ev.min()

    W, B = np.meshgrid(np.linspace(-1, 7, 240), np.linspace(-6, 16, 240))
    Z = np.mean((W[None] * x[:, None, None] + B[None] - y[:, None, None]) ** 2, axis=0)

    lr, n_step = 0.0155, 1500
    w, b = -0.5, 14.0
    path = [(w, b)]
    for _ in range(n_step):
        e = w * x + b - y
        w -= lr * 2 * (e * x).mean()
        b -= lr * 2 * e.mean()
        path.append((w, b))
    path = np.array(path)

    fig = plt.figure(figsize=(12.8, 4.9))

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(W, B, Z, cmap="viridis", alpha=.92, linewidth=0,
                     antialiased=True, rstride=4, cstride=4)
    ax1.set_xlabel("w", labelpad=-1); ax1.set_ylabel("b", labelpad=-1)
    ax1.set_zlabel("MSE", labelpad=6)
    ax1.tick_params(labelsize=7.5, pad=1)
    ax1.set_title("Mặt mất mát MSE(w, b): một cái bát LỒI\n"
                  "thả bi từ đâu cũng lăn về đúng một đáy", fontsize=10, pad=-4)
    ax1.view_init(elev=34, azim=-128)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.contour(W, B, Z, levels=np.geomspace(Z.min() + 1, Z.max(), 16),
                colors="0.6", linewidths=.8)
    ax2.plot(path[:50, 0], path[:50, 1], "-", color=C2, lw=2.4, zorder=4,
             label="50 bước đầu: LAO vào khe")
    ax2.plot(path[50:, 0], path[50:, 1], "-", color=C1, lw=2.4, zorder=4,
             label=f"{n_step - 50} bước sau: BÒ dọc khe")
    ax2.scatter([path[0, 0]], [path[0, 1]], s=95, color="k", zorder=6,
                label="điểm khởi tạo")
    ax2.scatter([3], [5], marker="*", s=340, color=C4, zorder=6, edgecolor="k",
                linewidth=.7, label="nghiệm tối ưu (3, 5)")
    ax2.annotate("cái KHE hẹp: contour dẹt\n"
                 f"số điều kiện $\\kappa \\approx {kappa:.0f}$\n"
                 f"→ cần tới {n_step} bước mới tới đáy",
                 xy=(2.3, 9.4), xytext=(3.35, 13.2), fontsize=8.6,
                 arrowprops=dict(arrowstyle="->", color="k", lw=1.4),
                 bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                           alpha=.94, edgecolor="0.8"))
    ax2.set_xlim(-1, 7); ax2.set_ylim(-6, 16)
    ax2.set_xlabel("w"); ax2.set_ylabel("b")
    ax2.legend(fontsize=7.8, loc="lower left", framealpha=.94)
    ax2.set_title("Nhìn từ trên xuống: GD luôn đi vuông góc với đường đồng mức",
                  fontsize=10)
    fig.suptitle("MSE của hồi quy tuyến tính là hàm LỒI: chỉ có MỘT cực tiểu toàn cục",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, .94])
    save(fig, "03_mat_mat_mat_va_duong_di_gd.png")


# ---------------------------------------------------------------- 4
def fig_learning_rate():
    """Ba chế độ của learning rate. Dùng feature ĐÃ CHUẨN HOÁ nên Hessian = 2I,
    ngưỡng phân kỳ lý thuyết đúng bằng lr = 2/lambda_max = 1.0 (kiểm chứng được)."""
    rng = np.random.default_rng(42)
    N = 60
    xr = np.linspace(0, 10, N)
    x = (xr - xr.mean()) / xr.std()
    y = 3 * xr + 5 + rng.normal(0, 1.5, N)
    w_opt = (x * y).mean() / (x * x).mean()
    b_opt = y.mean()

    def run(lr, steps=40):
        w, b = -3.0, 40.0
        P, L = [(w, b)], []
        for _ in range(steps):
            e = w * x + b - y
            L.append((e ** 2).mean())
            gw, gb = 2 * (e * x).mean(), 2 * e.mean()
            w -= lr * gw; b -= lr * gb
            if not np.isfinite(w) or abs(w) > 1e8:
                break
            P.append((w, b))
        return np.array(P), np.array(L)

    W, B = np.meshgrid(np.linspace(-14, 30, 300), np.linspace(-6, 46, 300))
    Z = np.mean((W[None] * x[:, None, None] + B[None] - y[:, None, None]) ** 2, axis=0)

    cfg = [(0.02, "lr QUÁ NHỎ  (0.02)", "40 bước vẫn chưa tới đích\nĐúng hướng, nhưng phí thời gian", C1),
           (0.35, "lr VỪA  (0.35)", "Hội tụ đúng tâm sau ~15 bước", C3),
           (1.02, "lr QUÁ LỚN  (1.02)", "Vượt ngưỡng 1.0 → văng ra xa dần\nLoss tăng vọt rồi thành inf", C2)]

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.4),
                             gridspec_kw={"height_ratios": [1.45, 1]})
    for j, (lr, title, note, col) in enumerate(cfg):
        P, L = run(lr)
        a_ = axes[0, j]
        a_.contour(W, B, Z, levels=np.geomspace(Z.min() + 1, Z.max(), 14),
                   cmap="Greys", linewidths=.8, alpha=.75)
        inside = np.isfinite(P[:, 0]) & (np.abs(P[:, 0]) < 1e6)
        a_.plot(P[inside, 0], P[inside, 1], "o-", color=col, ms=4, lw=1.6, zorder=4)
        a_.scatter([P[0, 0]], [P[0, 1]], s=70, color="k", zorder=6)
        a_.text(P[0, 0] + .8, P[0, 1] + .6, "khởi tạo", fontsize=8.4, zorder=6)
        a_.scatter([w_opt], [b_opt], marker="*", s=300, color=C4, edgecolor="k",
                   linewidth=.7, zorder=6)
        a_.annotate("nghiệm tối ưu", xy=(w_opt, b_opt), xycoords="data",
                    xytext=(.70, .90), textcoords="axes fraction", fontsize=8.6,
                    color=C4, fontweight="bold", zorder=7,
                    arrowprops=dict(arrowstyle="->", color=C4, lw=1.4),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              alpha=.92, edgecolor="0.8"))
        a_.set_xlim(-14, 30); a_.set_ylim(-6, 46)
        a_.set_xlabel("w"); a_.set_ylabel("b")
        a_.set_title(title, fontsize=10.5, fontweight="bold", color=col)
        a_.text(.03, .04, note, transform=a_.transAxes, fontsize=8.5, va="bottom",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=.92,
                          edgecolor="0.8"))
        b_ = axes[1, j]
        b_.plot(range(1, len(L) + 1), L, color=col, lw=2.2)
        b_.set_yscale("log"); b_.set_xlabel("bước"); b_.set_ylabel("MSE (thang log)")
        b_.set_ylim(.5, 3e6)
        b_.text(.97, .93, f"MSE cuối = {L[-1]:.4g}", transform=b_.transAxes,
                fontsize=8.8, ha="right", va="top", color=col, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=.92,
                          edgecolor="0.8"))
    fig.suptitle("Learning rate quyết định số phận của Gradient Descent  "
                 r"(ở đây feature đã chuẩn hoá nên ngưỡng phân kỳ đúng bằng $2/\lambda_{max}=1.0$)",
                 fontweight="bold", fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, .96])
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
        ("LỖI: quan hệ phi tuyến còn sót\n→ thêm $x^2$ hoặc đổi model", 0.32 * x ** 2 + rng.normal(0, 1.2, n), C4),
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
        lo, hi = r.min(), r.max()
        ax.set_ylim(lo - (hi - lo) * .10, hi + (hi - lo) * .18)
    fig.suptitle("Residual plot — công cụ chẩn đoán số 1 của hồi quy tuyến tính",
                 fontweight="bold")
    fig.tight_layout()
    save(fig, "06_chan_doan_phan_du.png")


# ---------------------------------------------------------------- 7
def fig_bias_variance():
    """Chú ý: PHẢI có StandardScaler trong pipeline. Không có nó, ma trận Vandermonde
    của PolynomialFeatures bậc cao bị suy biến số học và train RMSE lại TĂNG -
    một hiện tượng số học, không phải hiện tượng thống kê."""
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.metrics import mean_squared_error

    def poly(d):
        return make_pipeline(PolynomialFeatures(d), StandardScaler(), LinearRegression())

    rng = np.random.default_rng(1)
    nt = 18
    xt = np.sort(rng.uniform(0, 2 * np.pi, nt)); yt = np.sin(xt) + rng.normal(0, .22, nt)
    xv = np.sort(rng.uniform(0, 2 * np.pi, 300)); yv = np.sin(xv) + rng.normal(0, .22, 300)
    xx = np.linspace(0, 2 * np.pi, 500).reshape(-1, 1)

    fig, axes = plt.subplots(1, 4, figsize=(16.5, 3.9))
    for ax, (d, lab, col) in zip(axes[:3], [(1, "degree 1 — UNDERFIT (bias cao)", C1),
                                            (6, "degree 6 — VỪA ĐẸP", C3),
                                            (17, "degree 17 — OVERFIT (variance cao)", C2)]):
        m = poly(d).fit(xt.reshape(-1, 1), yt)
        ax.plot(xx, np.sin(xx), "--", color="gray", lw=1.6, label="hàm thật sin(x)")
        ax.plot(xx, m.predict(xx), color=col, lw=2.3, label=f"model bậc {d}")
        ax.scatter(xt, yt, s=34, color="k", alpha=.75, zorder=5,
                   label=f"train ({nt} điểm)")
        ax.set_ylim(-2.4, 2.4); ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_title(lab, fontsize=10)
        ax.legend(fontsize=7.4, loc="lower left", framealpha=.92)

    degs = list(range(1, 18))
    tr = [np.sqrt(mean_squared_error(yt, poly(d).fit(xt.reshape(-1, 1), yt)
                                     .predict(xt.reshape(-1, 1)))) for d in degs]
    te = [np.sqrt(mean_squared_error(yv, poly(d).fit(xt.reshape(-1, 1), yt)
                                     .predict(xv.reshape(-1, 1)))) for d in degs]
    ax = axes[3]
    ax.plot(degs, tr, "o-", color=C1, ms=4, label="train RMSE")
    ax.plot(degs, te, "s-", color=C2, ms=4, label="test RMSE")
    best = degs[int(np.argmin(te))]
    ax.axvline(best, color=C3, ls="--", lw=1.6)
    ax.set_yscale("log")
    ax.set_ylim(min(tr) * .72, max(te) * 2.4)
    ax.annotate(f"điểm ngọt\nbậc {best}", xy=(best, min(te)),
                xytext=(.56, .40), textcoords="axes fraction", fontsize=8.8,
                color=C3, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C3, lw=1.4),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=.93, edgecolor="0.8"))
    ax.set_xlabel("bậc đa thức (độ phức tạp)"); ax.set_ylabel("RMSE (thang log)")
    ax.set_title("Đường cong bias–variance", fontsize=10)
    ax.legend(fontsize=8, loc="upper left", framealpha=.92)
    fig.suptitle("Train RMSE giảm rồi nằm phẳng, Test RMSE giảm rồi TĂNG — đó là overfitting",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, .95])
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
