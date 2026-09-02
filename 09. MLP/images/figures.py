"""
Sinh toàn bộ hình minh hoạ cho Lab 09 - Multi-Layer Perceptron (MLP).

Chạy:  python figures.py
Kết quả: các file .png trong cùng thư mục, được nhúng vào notebook bằng markdown.

Lưu ý: script này không dùng PyTorch. Toàn bộ mạng nơ-ron ở đây được cài bằng
numpy thuần (lớp NPMLP bên dưới): vừa để hình luôn tái lập được, vừa để sinh viên
đọc thẳng phần forward/backward mà không bị framework che mất.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

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


# ==================================================================
#  Một MLP tí hon bằng numpy, dùng chung cho mọi hình bên dưới
# ==================================================================
def relu(z):
    return np.maximum(0, z)


def d_relu(z):
    return (z > 0).astype(float)


def tanh_(z):
    return np.tanh(z)


def d_tanh(z):
    return 1 - np.tanh(z) ** 2


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -60, 60)))


def d_sigmoid(z):
    s = sigmoid(z)
    return s * (1 - s)


ACT = {"relu": (relu, d_relu), "tanh": (tanh_, d_tanh), "sigmoid": (sigmoid, d_sigmoid)}


class NPMLP:
    """MLP phân loại nhị phân: [d_in, h1, ..., 1], output sigmoid + binary cross-entropy."""

    def __init__(self, sizes, act="tanh", seed=0, init="he"):
        self.sizes, self.actname = sizes, act
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            nin = sizes[i]
            if init == "he":
                s = np.sqrt(2.0 / nin)
            elif init == "xavier":
                s = np.sqrt(1.0 / nin)
            else:
                s = float(init)
            self.W.append(rng.normal(0, s, (nin, sizes[i + 1])))
            self.b.append(np.zeros(sizes[i + 1]))

    def forward(self, X, keep_cache=False):
        f, _ = ACT[self.actname]
        a = X
        cache = [(None, X)]
        L = len(self.W)
        for l in range(L):
            z = a @ self.W[l] + self.b[l]
            a = sigmoid(z) if l == L - 1 else f(z)
            cache.append((z, a))
        return (a, cache) if keep_cache else a

    def loss(self, X, y):
        p = np.clip(self.forward(X).ravel(), 1e-9, 1 - 1e-9)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def grads(self, X, y):
        _, df = ACT[self.actname]
        n = len(X)
        p, cache = self.forward(X, keep_cache=True)
        L = len(self.W)
        # delta của tầng output: sigmoid + BCE rút gọn thành (p - y)
        delta = (p - y.reshape(-1, 1)) / n
        gW, gb = [None] * L, [None] * L
        for l in range(L - 1, -1, -1):
            a_prev = cache[l][1]
            gW[l] = a_prev.T @ delta
            gb[l] = delta.sum(0)
            if l > 0:
                delta = (delta @ self.W[l].T) * df(cache[l][0])
        return gW, gb

    def step(self, X, y, lr):
        gW, gb = self.grads(X, y)
        for l in range(len(self.W)):
            self.W[l] -= lr * gW[l]
            self.b[l] -= lr * gb[l]

    def fit(self, X, y, lr=0.5, epochs=500, Xv=None, yv=None, snap_at=()):
        hist_tr, hist_va, snaps = [], [], {}
        for e in range(epochs + 1):
            if e in snap_at:
                snaps[e] = ([w.copy() for w in self.W], [b.copy() for b in self.b])
            hist_tr.append(self.loss(X, y))
            if Xv is not None:
                hist_va.append(self.loss(Xv, yv))
            if e < epochs:
                self.step(X, y, lr)
        return np.array(hist_tr), np.array(hist_va), snaps

    def acc(self, X, y):
        return float(((self.forward(X).ravel() > .5) == (y > .5)).mean())


def decision_grid(model, X, pad=.6, n=220):
    gx = np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, n)
    gy = np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, n)
    GX, GY = np.meshgrid(gx, gy)
    Z = model.forward(np.c_[GX.ravel(), GY.ravel()]).reshape(GX.shape)
    return GX, GY, Z


def draw_boundary(ax, model, X, y, title):
    GX, GY, Z = decision_grid(model, X)
    ax.contourf(GX, GY, Z, levels=np.linspace(0, 1, 21), cmap="RdBu_r", alpha=.55)
    ax.contour(GX, GY, Z, levels=[.5], colors="k", linewidths=1.8)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=14, color=C1, edgecolor="w", linewidth=.3)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=14, color=C2, edgecolor="w", linewidth=.3)
    ax.set_title(title, fontsize=9.5)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)


# ---------------------------------------------------------------- 1
def fig_xor_hidden_space():
    X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    y = np.array([0., 1., 1., 0.])

    # huấn luyện mạng 2-2-1 bằng tanh cho tới khi giải được XOR với loss đủ nhỏ
    net = None
    for seed in range(40):
        cand = NPMLP([2, 2, 1], act="tanh", seed=seed, init="xavier")
        cand.fit(X, y, lr=1.0, epochs=12000)
        if cand.acc(X, y) == 1.0 and cand.loss(X, y) < 0.02:
            net = cand
            break
    assert net is not None, "không tìm được nghiệm XOR"

    fig = plt.figure(figsize=(14.5, 4.6))

    # (a) XOR trong không gian gốc + vài đường thẳng thử
    ax = fig.add_subplot(1, 3, 1)
    for (a, b, c, col) in [(1, 1, -0.5, "0.4"), (1, 1, -1.5, "0.4"),
                           (1, -1, 0.5, "0.55"), (1, -1, -0.5, "0.55"),
                           (0, 1, -0.5, "0.7")]:
        xs = np.linspace(-.6, 1.6, 10)
        if b != 0:
            ax.plot(xs, (c * -1 - a * xs) / b if False else (-a * xs - c) / b,
                    ls="--", lw=1.3, color=col)
        else:
            ax.axvline(-c / a, ls="--", lw=1.3, color=col)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=260, color=C1, zorder=5,
               edgecolor="k", label="lớp 0")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=260, marker="s", color=C2, zorder=5,
               edgecolor="k", label="lớp 1")
    ax.set_xlim(-.6, 1.7); ax.set_ylim(-.6, 1.7)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.legend(fontsize=8, loc="upper left")
    # nhận xét đưa lên tiêu đề: trong trục không còn chỗ trống nào không có đường đứt đi qua
    ax.set_title("(a) XOR trong không gian gốc\nkhông đường thẳng nào tách đúng cả bốn điểm",
                 fontsize=9.5)

    # (b) không gian ẩn
    f, _ = ACT["tanh"]
    H = f(X @ net.W[0] + net.b[0])
    ax = fig.add_subplot(1, 3, 2)
    hx = np.linspace(H[:, 0].min() - .35, H[:, 0].max() + .35, 200)
    hy = np.linspace(H[:, 1].min() - .35, H[:, 1].max() + .35, 200)
    HX, HY = np.meshgrid(hx, hy)
    P = sigmoid(np.c_[HX.ravel(), HY.ravel()] @ net.W[1] + net.b[1]).reshape(HX.shape)
    ax.contourf(HX, HY, P, levels=np.linspace(0, 1, 21), cmap="RdBu_r", alpha=.55)
    ax.contour(HX, HY, P, levels=[.5], colors="k", linewidths=2.2)
    ax.scatter(H[y == 0, 0], H[y == 0, 1], s=260, color=C1, zorder=5, edgecolor="k")
    ax.scatter(H[y == 1, 0], H[y == 1, 1], s=260, marker="s", color=C2, zorder=5, edgecolor="k")
    dy = [.05, .05, .05, -.16]   # (1,1) trùng chỗ với (0,0) nên đẩy nhãn xuống
    for (px, py), (ox, oy), d in zip(H, X, dy):
        ax.text(px + .05, py + d, f"({ox:.0f},{oy:.0f})", fontsize=9)
    ax.set_xlim(hx[0], hx[-1] + .55)      # chừa chỗ cho nhãn của 2 điểm bên phải
    ax.set_xlabel("$h_1$"); ax.set_ylabel("$h_2$"); ax.grid(False)
    ax.set_title("(b) Bốn điểm đó trong không gian ẩn $(h_1,h_2)$", fontsize=9.5)
    ax.text(.03, .97, "hai điểm lớp 0 bị đẩy trùng nhau,\ngiờ tách được bằng một đường thẳng",
            transform=ax.transAxes, fontsize=8.5, va="top", color="#065f46",
            bbox=dict(fc="white", ec="0.7", alpha=.95, boxstyle="round,pad=0.28"))

    # (c) ranh giới mạng vẽ ngược lại trong không gian gốc
    ax = fig.add_subplot(1, 3, 3)
    gx = np.linspace(-1.4, 2.4, 320); gy = np.linspace(-1.4, 2.4, 320)
    GX, GY = np.meshgrid(gx, gy)
    Z = net.forward(np.c_[GX.ravel(), GY.ravel()]).reshape(GX.shape)
    ax.contourf(GX, GY, Z, levels=np.linspace(0, 1, 21), cmap="RdBu_r", alpha=.6)
    ax.contour(GX, GY, Z, levels=[.5], colors="k", linewidths=2)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=260, color=C1, zorder=5, edgecolor="k")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=260, marker="s", color=C2, zorder=5, edgecolor="k")
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.grid(False)
    # góc trên-trái là vùng đỏ thuần, không có đường biên đi qua, nên đặt chú thích ở đó
    ax.text(.03, .97, "vùng xanh (lớp 0) là một dải kẹp giữa\nhai đường thẳng, trong khi perceptron\n"
            "đơn chỉ tạo được một nửa mặt phẳng\nnên không bao giờ vẽ nổi dải này",
            transform=ax.transAxes, fontsize=8.5, color="k", va="top",
            bbox=dict(fc="white", ec="0.6", alpha=.95, boxstyle="round,pad=0.3"))
    ax.set_title("(c) Ranh giới đó vẽ lại trong không gian gốc", fontsize=9.5)

    fig.suptitle("Tầng ẩn như một phép biến đổi không gian",
                 fontweight="bold", fontsize=11.5)
    fig.text(.5, -.03, "Mạng ở đây là 2-2-1, tầng ẩn dùng tanh. Mạng không hề 'vẽ đường cong': "
             "nó bẻ cong không gian rồi vẫn cắt bằng một đường thẳng.",
             ha="center", va="top", fontsize=10, color="0.25")
    fig.tight_layout()
    save(fig, "01_xor_khong_gian_an.png")


# ---------------------------------------------------------------- 2
def fig_architecture():
    fig, ax = plt.subplots(figsize=(13, 6.6))
    layers = [("Input $x$", 784, 6), ("Hidden 1 (ReLU)", 256, 6),
              ("Hidden 2 (ReLU)", 128, 6), ("Output (logits)", 10, 4)]
    xs = [0, 3.2, 6.4, 9.6]
    pos = []
    for (name, n, shown), x in zip(layers, xs):
        ys = np.linspace(2.6, -2.6, shown)
        pos.append(ys)
    cols = [C1, C3, C3, C4]
    for li in range(len(layers) - 1):
        for ya in pos[li]:
            for yb in pos[li + 1]:
                ax.plot([xs[li], xs[li + 1]], [ya, yb], color="0.82", lw=.6, zorder=1)
    for li, ((name, n, shown), x) in enumerate(zip(layers, xs)):
        for yy in pos[li]:
            ax.add_patch(Circle((x, yy), .22, color=cols[li], zorder=3, ec="k", lw=.8))
        # tên tầng và số nút đặt ngay dưới cột nút, phía trên dành riêng cho hộp tham số
        ax.text(x, -3.1, f"{name}\n{n} nút", ha="center", va="top", fontsize=10.5,
                color=cols[li], fontweight="bold")
        ax.text(x, 0, "⋮", ha="center", va="center", fontsize=20, zorder=4)

    # hộp tham số + công thức đặt phía trên hàng nút cao nhất (y = 2.6), không đè lên đường nối
    info = [(1.6, r"$h_1=\mathrm{ReLU}(W_1^{\top}x+b_1)$" "\n"
             r"$W_1 \in \mathbb{R}^{784\times256},\ b_1 \in \mathbb{R}^{256}$" "\n"
             "200 704 + 256 = 200 960 tham số"),
            (4.8, r"$h_2=\mathrm{ReLU}(W_2^{\top}h_1+b_2)$" "\n"
             r"$W_2 \in \mathbb{R}^{256\times128},\ b_2 \in \mathbb{R}^{128}$" "\n"
             "32 768 + 128 = 32 896 tham số"),
            (8.0, r"$z=W_3^{\top}h_2+b_3$" "\n"
             r"$W_3 \in \mathbb{R}^{128\times10},\ b_3 \in \mathbb{R}^{10}$" "\n"
             "1 280 + 10 = 1 290 tham số")]
    for x, t in info:
        ax.text(x, 3.05, t, ha="center", va="bottom", fontsize=9.5,
                bbox=dict(fc="#f8fafc", ec="0.6", boxstyle="round,pad=0.4"))
    ax.text(4.8, -4.6, va="top",
            s="Tổng: 200 960 + 32 896 + 1 290 = 235 146 tham số\n"
            r"Công thức: mỗi tầng Linear($n_{in}\to n_{out}$) có $n_{in}\,n_{out}$ trọng số + $n_{out}$ bias",
            ha="center", fontsize=10.5,
            bbox=dict(fc="#fffbeb", ec=C4, boxstyle="round,pad=0.45"))
    ax.text(4.8, 4.85, "Mỗi nút nối với toàn bộ nút của tầng trước: tầng 'fully connected'",
            ha="center", va="center", fontsize=11, color="0.3")
    ax.set_xlim(-1.2, 11); ax.set_ylim(-6.3, 5.3)
    ax.axis("off")
    ax.set_title("Kiến trúc MLP dùng trong lab: 784-256-128-10",
                 fontweight="bold", fontsize=12)
    save(fig, "02_kien_truc_mang.png")


# ---------------------------------------------------------------- 3
def fig_activations():
    from scipy.special import erf
    z = np.linspace(-6, 6, 600)
    Phi = .5 * (1 + erf(z / np.sqrt(2)))
    phi = np.exp(-z ** 2 / 2) / np.sqrt(2 * np.pi)
    fns = [
        ("Sigmoid", sigmoid(z), sigmoid(z) * (1 - sigmoid(z)),
         r"$\frac{1}{1+e^{-z}}$", "bão hoà 2 đầu, $\\phi'\\leq 0.25$;\nvùng tô đỏ có $\\phi'\\approx 0$\nnên gradient tắt"),
        ("Tanh", np.tanh(z), 1 - np.tanh(z) ** 2,
         r"$\tanh z$", "bão hoà 2 đầu\nnhưng zero-centered;\nvùng tô đỏ gradient tắt"),
        ("ReLU", np.maximum(0, z), (z > 0).astype(float),
         r"$\max(0,z)$", "không bão hoà bên phải\nnhưng 'chết' bên trái"),
        ("LeakyReLU", np.where(z > 0, z, .1 * z), np.where(z > 0, 1., .1),
         r"$\max(0.1z,\,z)$", "sửa lỗi neuron chết"),
        ("GELU", z * Phi, Phi + z * phi,
         r"$z\,\Phi(z)$", "trơn, dùng trong\nTransformer"),
    ]
    fig, axes = plt.subplots(2, 5, figsize=(16, 6.2), sharex=True)
    for j, (name, v, dv, formula, note) in enumerate(fns):
        ax = axes[0, j]
        ax.plot(z, v, color=C1, lw=2.2)
        ax.axhline(0, color="k", lw=.7); ax.axvline(0, color="k", lw=.7)
        ax.set_title(f"{name}\n{formula}", fontsize=11)
        ax.set_ylim(-1.6, 3.2)
        if j == 0:
            ax.set_ylabel(r"$\phi(z)$", fontsize=12)
        ax = axes[1, j]
        ax.plot(z, dv, color=C2, lw=2.2)
        ax.axhline(0, color="k", lw=.7); ax.axvline(0, color="k", lw=.7)
        ax.set_ylim(-.15, 1.7)              # dải trống phía trên để hộp chú thích không chạm đỉnh đạo hàm
        ax.set_xlabel("z")
        if j == 0:
            ax.set_ylabel(r"$\phi'(z)$", fontsize=12)
        ax.text(.03, .93, note, transform=ax.transAxes, fontsize=8.5, va="top",
                color="#7f1d1d" if j < 2 else "#065f46",
                bbox=dict(fc="white", ec="0.75", alpha=.9, boxstyle="round,pad=0.25"))
        if j < 2:
            ax.axvspan(-6, -2.5, color=C2, alpha=.10)
            ax.axvspan(2.5, 6, color=C2, alpha=.10)
    fig.suptitle("Các hàm kích hoạt và đạo hàm của chúng",
                 fontweight="bold", fontsize=12)
    fig.text(.5, -.02, r"Hàng trên là $\phi(z)$, hàng dưới là $\phi'(z)$: chính đạo hàm mới là "
             "thứ thực sự chạy trong backprop.\nĐạo hàm nhỏ ở đâu thì gradient chết ở đó.",
             ha="center", va="top", fontsize=10.5, color="0.25")
    fig.tight_layout()
    save(fig, "03_ham_kich_hoat.png")


# ---------------------------------------------------------------- 4
def fig_vanishing_gradient():
    rng = np.random.default_rng(0)
    n, d, L = 256, 64, 10
    X = rng.normal(0, 1, (n, d))
    y = (rng.random(n) > .5).astype(float)

    res = {}
    for act, init, col in [("sigmoid", "xavier", C2), ("tanh", "xavier", C4), ("relu", "he", C3)]:
        net = NPMLP([d] + [64] * L + [1], act=act, seed=1, init=init)
        gW, _ = net.grads(X, y)
        res[act] = ([np.abs(g).mean() for g in gW[:-1]], col)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    for act, (g, col) in res.items():
        ax.plot(range(1, L + 1), g, "o-", color=col, lw=2, ms=6,
                label=f"{act} ({'He' if act == 'relu' else 'Xavier'} init)")
    ax.set_yscale("log")
    ax.set_xlabel("tầng thứ $l$  (1 = gần input nhất, 10 = gần output nhất)")
    ax.set_ylabel(r"độ lớn gradient trung bình  $\overline{|\partial L/\partial W^{(l)}|}$  (log)")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 300)                 # chừa dải trống phía trên cho legend
    ax.legend(fontsize=9, loc="upper right", framealpha=.95,
              title="mạng 10 tầng ẩn, 64 nút mỗi tầng", title_fontsize=8.5)
    ax.set_title("(a) Gradient theo tầng, đo lúc mới khởi tạo", fontsize=10.5)
    r = res["sigmoid"][0]
    ratio = f"{r[-1] / r[0]:,.0f}".replace(",", ".")
    # chú thích đặt ngay trên đoạn phẳng của đường sigmoid (tầng 1-3), không cần đường dẫn
    ax.text(1.0, r[0] * 40, f"gradient ở tầng 1 nhỏ hơn tầng 10\nkhoảng {ratio} lần, nên tầng đầu\ngần như không học được gì",
            ha="left", va="bottom", fontsize=9, color=C2,
            bbox=dict(fc="white", ec="0.8", alpha=.92, boxstyle="round,pad=0.3"))

    ax = axes[1]
    k = np.arange(1, 13)
    for m, col, lab in [(0.25, C2, r"$\phi'_{\max}=0.25$ (sigmoid): tắt dần"),
                        (1.0, C3, r"$\phi'=1$ (ReLU vùng dương): giữ nguyên"),
                        (1.5, C1, r"$\phi'\cdot\|W\|=1.5$: tăng theo cấp số nhân")]:
        ax.plot(k, m ** k, "o-", color=col, lw=2, ms=5, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("số tầng phải đi ngược qua")
    ax.set_ylabel("hệ số nhân tích luỹ (log)")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 200)                 # chừa dải trống phía trên cho legend
    ax.legend(fontsize=8.5, loc="upper left", framealpha=.95)
    ax.set_title(r"(b) Hệ số nhân tích luỹ $\prod_l \phi'(z^{(l)})\,W^{(l)}$", fontsize=10.5)
    fig.suptitle("Vanishing gradient trong mạng sâu", fontweight="bold", fontsize=12)
    fig.text(.5, -.02, "Gốc rễ toán học: gradient là một tích chạy dọc các tầng. "
             "Nhân nhiều số nhỏ hơn 1 thì về 0, nhân nhiều số\nlớn hơn 1 thì tăng vọt. "
             "Đó là lý do mạng sâu dùng sigmoid gần như không học được tầng đầu.",
             ha="center", va="top", fontsize=10, color="0.25")
    fig.tight_layout()
    save(fig, "04_vanishing_gradient.png")


# ---------------------------------------------------------------- 5
def fig_backprop_graph():
    fig, ax = plt.subplots(figsize=(14.5, 10.2))
    ax.axis("off")

    boxes = [
        (0.6, 3.6, "$x_1$"), (0.6, 2.2, "$x_2$"),
        (3.0, 4.3, "$z^{(1)}_1$"), (3.0, 1.5, "$z^{(1)}_2$"),
        (5.0, 4.3, "$a^{(1)}_1$"), (5.0, 1.5, "$a^{(1)}_2$"),
        (7.4, 2.9, "$z^{(2)}$"), (9.4, 2.9, "$\\hat{y}$"), (11.6, 2.9, "$L$"),
    ]
    coord = {}
    for (x, y, t) in boxes:
        c = C1 if t.startswith("$x") else (C4 if "L" in t else C3)
        ax.add_patch(Rectangle((x - .42, y - .34), .84, .68, fc="white", ec=c, lw=2,
                               zorder=3, joinstyle="round"))
        ax.text(x, y, t, ha="center", va="center", fontsize=13, zorder=4)
        coord[t] = (x, y)

    edges = [("$x_1$", "$z^{(1)}_1$"), ("$x_1$", "$z^{(1)}_2$"),
             ("$x_2$", "$z^{(1)}_1$"), ("$x_2$", "$z^{(1)}_2$"),
             ("$z^{(1)}_1$", "$a^{(1)}_1$"), ("$z^{(1)}_2$", "$a^{(1)}_2$"),
             ("$a^{(1)}_1$", "$z^{(2)}$"), ("$a^{(1)}_2$", "$z^{(2)}$"),
             ("$z^{(2)}$", "$\\hat{y}$"), ("$\\hat{y}$", "$L$")]
    for a, b in edges:
        (xa, ya), (xb, yb) = coord[a], coord[b]
        ax.add_patch(FancyArrowPatch((xa + .45, ya), (xb - .45, yb),
                                     arrowstyle="-|>", mutation_scale=15,
                                     color=C1, lw=1.8, alpha=.9, zorder=2))
        ax.add_patch(FancyArrowPatch((xb - .45, yb - .5), (xa + .45, ya - .5),
                                     arrowstyle="-|>", mutation_scale=15,
                                     color=C2, lw=1.5, alpha=.85, ls="--", zorder=2,
                                     connectionstyle="arc3,rad=0.14"))

    ax.text(0.0, 5.9, "Forward (nét liền, xanh): tính giá trị", color=C1,
            fontsize=12.5, fontweight="bold")
    ax.text(0.0, 5.35, "Backward (nét đứt, đỏ): tính đạo hàm", color=C2,
            fontsize=12.5, fontweight="bold")
    ax.text(7.4, 5.9, "Mạng 2-2-1, một mẫu, output sigmoid + binary cross-entropy",
            fontsize=11, color="0.3")

    fwd = ("Forward: đi từ trái sang phải, lưu lại mọi giá trị trung gian\n"
           r"$z^{(1)} = W^{(1)\top}x + b^{(1)}$" "\n"
           r"$a^{(1)} = \phi(z^{(1)})$" "\n"
           r"$z^{(2)} = W^{(2)\top}a^{(1)} + b^{(2)}$" "\n"
           r"$\hat{y} = \sigma(z^{(2)})$" "\n"
           r"$L = -[\,y\log\hat{y} + (1-y)\log(1-\hat{y})\,]$")
    ax.text(0.2, .35, fwd, va="top", ha="left", fontsize=12, color=C1, linespacing=1.9,
            bbox=dict(fc="#eff6ff", ec=C1, boxstyle="round,pad=0.55"))

    bwd = ("Backward: đi ngược lại, dùng lại các giá trị vừa lưu\n"
           r"$\dfrac{\partial L}{\partial \hat{y}} = \dfrac{\hat{y}-y}{\hat{y}(1-\hat{y})}$"
           "        " r"$\delta^{(2)} \equiv \dfrac{\partial L}{\partial z^{(2)}} = \hat{y}-y$" "\n"
           r"$\nabla_{W^{(2)}}L = a^{(1)}\,\delta^{(2)\top}$"
           "        " r"$\nabla_{b^{(2)}}L = \delta^{(2)}$" "\n"
           r"$\delta^{(1)} = (W^{(2)}\delta^{(2)})\odot\phi'(z^{(1)})$" "\n"
           r"$\nabla_{W^{(1)}}L = x\,\delta^{(1)\top}$"
           "        " r"$\nabla_{b^{(1)}}L = \delta^{(1)}$")
    ax.text(6.6, .35, bwd, va="top", ha="left", fontsize=12, color=C2, linespacing=2.4,
            bbox=dict(fc="#fef2f2", ec=C2, boxstyle="round,pad=0.55"))

    ax.text(6.3, -4.85,
            "Một phép rút gọn quan trọng: sigmoid (hoặc softmax) + cross-entropy "
            r"$\Rightarrow\ \delta^{(L)} = \hat{y} - y$" "\n"
            r"Đạo hàm $\sigma'(z)=\sigma(1-\sigma)$ ở tử số triệt tiêu đúng mẫu số của loss, "
            "nên gradient không bị bão hoà ngay cả khi $\\hat{y}$ sai nhiều.\n"
            "Đó cũng là lý do PyTorch gộp softmax vào trong nn.CrossEntropyLoss thay vì để bạn tự viết.",
            ha="center", va="top", fontsize=11, color="#065f46", linespacing=1.7,
            bbox=dict(fc="#ecfdf5", ec=C3, boxstyle="round,pad=0.5"))

    ax.text(6.3, -3.55,
            "Backprop không phải là thuật toán học. Nó chỉ là quy tắc chuỗi được sắp xếp lại\n"
            "sao cho mỗi đạo hàm trung gian chỉ phải tính đúng một lần.",
            ha="center", va="top", fontsize=11, color="0.25", linespacing=1.5)
    ax.set_xlim(-.4, 12.9); ax.set_ylim(-7.4, 6.4)
    ax.set_title("Backpropagation trên đồ thị tính toán",
                 fontweight="bold", fontsize=13)
    save(fig, "05_backprop_do_thi.png")


# ---------------------------------------------------------------- 6
def fig_boundary_epochs():
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=400, noise=.18, random_state=0)
    X = (X - X.mean(0)) / X.std(0)
    y = y.astype(float)

    snap_at = (0, 20, 100, 1000)
    net = NPMLP([2, 16, 16, 1], act="tanh", seed=3, init="xavier")
    hist, _, snaps = net.fit(X, y, lr=1.5, epochs=1000, snap_at=snap_at)

    fig, axes = plt.subplots(1, 5, figsize=(17.5, 3.9))
    for ax, e in zip(axes, snap_at):
        tmp = NPMLP([2, 16, 16, 1], act="tanh", seed=3, init="xavier")
        tmp.W, tmp.b = snaps[e]
        draw_boundary(ax, tmp, X, y,
                      f"epoch {e}\nloss={hist[e]:.3f}  acc={tmp.acc(X, y) * 100:.1f}%")
    ax = axes[4]
    ax.plot(hist, color=C1, lw=2)
    # nhãn đặt lệch về phía không có đường: 0 và 100 sang phải, 20 xuống dưới-phải, 1000 lên trên
    off = {0: (9, 1), 20: (0, -17), 100: (11, 3), 1000: (-6, 13)}
    for e in snap_at:
        ax.scatter([e], [hist[e]], s=45, color=C2, zorder=5)
        ax.annotate(str(e), (e, hist[e]), textcoords="offset points",
                    xytext=off[e], fontsize=8.5, color=C2, zorder=6,
                    ha="center" if e == 20 else "left")
    ax.set_xlim(-len(hist) * .05, len(hist) * 1.10)
    ax.set_xlabel("epoch"); ax.set_ylabel("binary cross-entropy")
    ax.set_title("Đường loss tương ứng", fontsize=9.5)
    fig.suptitle("Ranh giới quyết định qua các epoch", fontweight="bold", fontsize=11.5)
    fig.text(.5, -.04, "Mạng 2-16-16-1, kích hoạt tanh, gradient descent full-batch, cài bằng numpy thuần. "
             "Ranh giới bắt đầu gần như\ntuyến tính, rồi cong dần cho tới khi ôm được hai trăng lưỡi liềm.",
             ha="center", va="top", fontsize=10, color="0.25")
    fig.tight_layout()
    save(fig, "06_ranh_gioi_theo_epoch.png")


# ---------------------------------------------------------------- 7
def fig_hidden_width():
    from sklearn.datasets import make_moons
    Xtr, ytr = make_moons(n_samples=120, noise=.30, random_state=1)
    Xte, yte = make_moons(n_samples=600, noise=.30, random_state=2)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ytr, yte = ytr.astype(float), yte.astype(float)

    widths = [1, 3, 8, 64]
    fig, axes = plt.subplots(1, 5, figsize=(17.5, 3.9))
    tr_l, te_l = [], []
    allw = [1, 2, 3, 5, 8, 16, 32, 64, 128]
    nets = {}
    for h in allw:
        net = NPMLP([2, h, 1], act="tanh", seed=0, init="xavier")
        net.fit(Xtr, ytr, lr=1.2, epochs=4000)
        tr_l.append(net.loss(Xtr, ytr)); te_l.append(net.loss(Xte, yte))
        nets[h] = net
    for ax, h in zip(axes, widths):
        net = nets[h]
        draw_boundary(ax, net, Xtr, ytr,
                      f"{h} neuron ẩn\ntrain acc={net.acc(Xtr, ytr) * 100:.0f}%  "
                      f"test acc={net.acc(Xte, yte) * 100:.0f}%")
    # đặt nhãn chẩn đoán ngay dưới khung để không đè lên điểm dữ liệu
    axes[0].set_xlabel("underfit: chỉ vẽ được một đường thẳng",
                       fontsize=9, color="#7f1d1d")
    axes[3].set_xlabel("overfit: ranh giới lượn theo từng điểm nhiễu",
                       fontsize=9, color="#7f1d1d")
    ax = axes[4]
    ax.plot(allw, tr_l, "o-", color=C1, label="train loss")
    ax.plot(allw, te_l, "s-", color=C2, label="test loss")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("số neuron ẩn (log)"); ax.set_ylabel("cross-entropy")
    ax.axvline(allw[int(np.argmin(te_l))], color=C3, ls="--", lw=1.6)
    ax.set_ylim(0, max(te_l) * 1.52)          # chừa dải trống phía trên cho legend
    ax.text(allw[int(np.argmin(te_l))] * 1.15, max(te_l) * .92, "điểm cân bằng", color=C3, fontsize=9,
            bbox=dict(fc="white", ec="none", alpha=.85, boxstyle="round,pad=0.2"))
    ax.legend(fontsize=8.5, loc="upper right", framealpha=.95)
    ax.set_title("Loss theo số neuron ẩn", fontsize=9.5)
    fig.suptitle("Số neuron ẩn và sức chứa của mạng", fontweight="bold", fontsize=11.5)
    fig.text(.5, -.10, "Dữ liệu moons nhiễu mạnh, chỉ 120 điểm train, không dùng regularization. "
             "Sức chứa quá ít thì underfit,\nquá nhiều thì overfit: train loss vẫn giảm đều "
             "trong khi test loss giảm rồi tăng trở lại.",
             ha="center", va="top", fontsize=10, color="0.25")
    fig.tight_layout()
    save(fig, "07_so_neuron_an.png")


# ---------------------------------------------------------------- 8
def fig_learning_rate():
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=300, noise=.15, random_state=0)
    X = (X - X.mean(0)) / X.std(0); y = y.astype(float)

    cfg = [(0.01, "lr = 0.01, quá nhỏ", C1, "giảm rất chậm, 300 epoch vẫn chưa học xong"),
           (1.0, "lr = 1, vừa", C3, "giảm nhanh, mượt, ổn định"),
           (3.0, "lr = 3, hơi lớn", C4, "vẫn giảm được nhưng dao động"),
           (10.0, "lr = 10, quá lớn", C2, "phân kỳ, loss tăng vọt rồi kẹt ở acc 50%")]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
    for lr, lab, col, note in cfg:
        net = NPMLP([2, 32, 32, 1], act="relu", seed=0, init="he")
        h, _, _ = net.fit(X, y, lr=lr, epochs=300)
        axes[0].plot(h, color=col, lw=2,
                     label=f"{lab}: {note}  [acc cuối {net.acc(X, y) * 100:.0f}%]")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss (thang log)")
    lo, hi = axes[0].get_ylim()
    axes[0].set_ylim(lo, hi * 60)             # chừa dải trống phía trên cho legend
    axes[0].legend(fontsize=8.2, loc="upper center", framealpha=.95,
                   title="mạng 2-32-32-1, ReLU, He init, cùng bộ trọng số ban đầu",
                   title_fontsize=8.2)
    axes[0].set_title("(a) Cùng một mạng, chỉ khác learning rate", fontsize=10.5)

    # minh hoạ trên mặt cắt 1 chiều
    ax = axes[1]
    w = np.linspace(-3, 3, 400)
    Lw = w ** 2
    ax.plot(w, Lw, color="0.6", lw=2)
    for lr, col, lab, w0 in [(0.05, C1, "lr nhỏ: bước rất ngắn", -2.6),
                             (0.45, C3, "lr vừa: tiến thẳng về đáy", -2.6),
                             (1.05, C2, "lr quá lớn: vượt qua đáy, nảy ra xa", -2.0)]:
        p = [w0]
        for _ in range(9 if lr < 1 else 5):
            p.append(p[-1] - lr * 2 * p[-1])
        p = np.array(p)
        ax.plot(p, p ** 2, "o-", color=col, ms=5, lw=1.5, label=lab, alpha=.9)
    ax.set_ylim(-.5, 18.5)                    # chừa dải trống phía trên cho legend và chú thích
    ax.legend(fontsize=8.5, loc="upper left", framealpha=.95)
    ax.set_xlabel("tham số $w$"); ax.set_ylabel("$L(w)=w^2$")
    ax.set_title(r"(b) Bước cập nhật $w \leftarrow w - \eta\,\nabla L$ trên $L(w)=w^2$",
                 fontsize=10.5)
    ax.text(.04, .74, r"$\eta$ quá lớn thì bước nhảy vượt qua đáy sang" "\n"
            "bờ bên kia, điểm mới còn cao hơn chỗ cũ.",
            transform=ax.transAxes, fontsize=8.5, va="top", color="0.2",
            bbox=dict(fc="white", ec="0.75", alpha=.92, boxstyle="round,pad=0.28"))
    fig.suptitle("Ảnh hưởng của learning rate tới huấn luyện",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    save(fig, "08_learning_rate.png")


# ---------------------------------------------------------------- 9
def fig_overfitting_dropout():
    from sklearn.datasets import make_moons
    Xtr, ytr = make_moons(n_samples=60, noise=.35, random_state=4)
    Xva, yva = make_moons(n_samples=600, noise=.35, random_state=5)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
    ytr, yva = ytr.astype(float), yva.astype(float)

    net = NPMLP([2, 64, 64, 1], act="tanh", seed=0, init="xavier")
    htr, hva, _ = net.fit(Xtr, ytr, lr=0.3, epochs=2500, Xv=Xva, yv=yva)
    best = int(np.argmin(hva))

    fig = plt.figure(figsize=(14.5, 5.2))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(htr, color=C1, lw=2, label="train loss (60 mẫu)")
    ax.plot(hva, color=C2, lw=2, label="validation loss (600 mẫu)")
    ax.axvline(best, color=C3, ls="--", lw=2)
    ax.scatter([best], [hva[best]], s=90, color=C3, zorder=6)
    ax.axvspan(best, len(htr), color=C2, alpha=.07)
    ax.set_ylim(0, max(hva) * 1.42)           # chừa dải trống phía trên cho chú thích
    # một hộp duy nhất ở dải trống phía trên, bên phải vạch đứng; điểm dừng đã có vạch và chấm đánh dấu
    ax.text(best + 90, max(hva) * 1.38,
            f"early stopping: dừng ở epoch {best}, val loss nhỏ nhất = {hva[best]:.3f}\n"
            "vùng tô hồng bên phải là vùng overfit: train loss vẫn giảm\n"
            "về 0 nhưng val loss tăng đều",
            ha="left", va="top", fontsize=9.5, color="0.2",
            bbox=dict(fc="white", ec="0.75", alpha=.95, boxstyle="round,pad=0.35"))
    ax.set_xlabel("epoch"); ax.set_ylabel("binary cross-entropy")
    # legend đặt ở khoảng trống giữa hai đường (phía phải, quanh y = 0.4)
    ax.legend(fontsize=9, loc="center right", bbox_to_anchor=(1.0, .40))
    ax.set_title("(a) Train loss và validation loss theo epoch", fontsize=10.5)

    # sơ đồ dropout
    ax = fig.add_subplot(1, 2, 2)
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(-4.1, 5.4)
    dropped = [0, 2, 4]        # neuron bị tắt ở tầng ẩn của panel phải
    for panel, (x0, title) in enumerate(
            [(0.4, "Không dropout\n(mọi neuron đều bật)"),
             (5.4, "Có dropout p=0.5 (chỉ lúc train)\nmỗi bước tắt ngẫu nhiên một nửa neuron")]):
        xs = [x0, x0 + 1.5, x0 + 3.0]
        ys = [np.linspace(3.4, 1.0, 3), np.linspace(4.2, .2, 5), np.linspace(3.0, 1.4, 2)]
        for li in range(2):
            for ai, ya in enumerate(ys[li]):
                for bi, yb in enumerate(ys[li + 1]):
                    dead = panel == 1 and ((li == 0 and bi in dropped) or
                                           (li == 1 and ai in dropped))
                    ax.plot([xs[li], xs[li + 1]], [ya, yb],
                            color="0.93" if dead else "0.72", lw=.8, zorder=1)
        for li, yy in enumerate(ys):
            for k, y_ in enumerate(yy):
                dead = panel == 1 and li == 1 and k in dropped
                ax.add_patch(Circle((xs[li], y_), .19, zorder=3, lw=1.2,
                                    fc="#f1f5f9" if dead else [C1, C3, C4][li],
                                    ec="0.55" if dead else "k"))
                if dead:
                    ax.plot([xs[li] - .13, xs[li] + .13], [y_ - .13, y_ + .13],
                            color=C2, lw=2.2, zorder=4)
                    ax.plot([xs[li] - .13, xs[li] + .13], [y_ + .13, y_ - .13],
                            color=C2, lw=2.2, zorder=4)
        ax.text(x0 + 1.5, 5.0, title, ha="center", fontsize=10, fontweight="bold")
    ax.text(5.0, -3.9,
            "Lúc inference phải bật lại toàn bộ neuron, nhưng khi đó tổng đầu vào của mỗi neuron\n"
            r"lớn gấp $1/(1-p)$ lần so với lúc train, nên phải bù lại. PyTorch dùng 'inverted dropout':"
            "\n"
            r"lúc train đã chia sẵn cho $(1-p)$, lúc eval không làm gì, vì vậy phải gọi "
            "model.eval() khi đánh giá;\nnếu quên thì dropout vẫn bật và kết quả đánh giá bị nhiễu ngẫu nhiên.",
            ha="center", va="bottom", fontsize=9.5,
            bbox=dict(fc="#fffbeb", ec=C4, boxstyle="round,pad=0.4"))
    ax.text(5.0, -.85, "Mỗi mini-batch tắt một tập neuron khác nhau, nên mạng không thể "
                     "phụ thuộc vào riêng một neuron nào\n(giống như huấn luyện một 'ensemble' "
                     "rất lớn các mạng con rồi lấy trung bình)",
            ha="center", va="center", fontsize=9.5, color="0.25")
    fig.suptitle("Overfitting, early stopping và dropout",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    save(fig, "09_overfitting_dropout.png")


# ---------------------------------------------------------------- 10
def fig_weight_init():
    rng = np.random.default_rng(0)
    n, d, L = 1000, 500, 6
    X = rng.normal(0, 1, (n, d))
    cfgs = [("Quá nhỏ:  $W\\sim N(0,\\,0.01^2)$", lambda nin: 0.01, C2,
             "activation co về 0 theo cấp số nhân,\ntín hiệu biến mất, gradient cũng biến mất"),
            ("Quá lớn:  $W\\sim N(0,\\,1^2)$", lambda nin: 1.0, C4,
             "tanh bão hoà ở ±1 nên $\\phi'\\approx 0$,\ngradient chết vì bão hoà"),
            ("Xavier:  $W\\sim N(0,\\,1/n_{in})$", lambda nin: np.sqrt(1.0 / nin), C3,
             "activation trải đều, không dồn về 0\ncũng không dán vào ±1; std co lại\nrất chậm nên tín hiệu và gradient\nđều còn ở mọi tầng")]

    fig, axes = plt.subplots(3, L, figsize=(16.5, 7.6))
    for r, (name, sfun, col, note) in enumerate(cfgs):
        a = X.copy()
        for l in range(L):
            W = rng.normal(0, sfun(a.shape[1]), (a.shape[1], d))
            a = np.tanh(a @ W)
            ax = axes[r, l]
            ax.hist(a.ravel(), bins=60, range=(-1, 1), color=col, alpha=.8)
            ax.set_yticks([]); ax.set_xlim(-1.05, 1.05)
            ax.set_title(f"tầng {l + 1}   std={a.std():.3f}", fontsize=9)
            if l == 0:
                ax.set_ylabel(name, fontsize=10)
            ax.grid(False)
        axes[r, L - 1].text(1.06, .5, note, transform=axes[r, L - 1].transAxes,
                            fontsize=8.5, va="center", color=col)
    fig.suptitle("Phân phối activation qua 6 tầng tanh, 500 nút mỗi tầng",
                 fontweight="bold", fontsize=11.5)
    fig.text(.44, -.02, "Ba hàng chạy trên cùng dữ liệu và cùng kiến trúc, chỉ khác nhau ở cách khởi tạo trọng số.\n"
             r"Xavier: $\mathrm{Var}(W)=1/n_{in}$ cho tanh/sigmoid;  "
             r"He: $\mathrm{Var}(W)=2/n_{in}$ cho ReLU (bù lại việc ReLU cắt bỏ một nửa tín hiệu).",
             ha="center", va="top", fontsize=10, color="0.25")
    fig.tight_layout(rect=[0, 0, .88, 1])
    save(fig, "10_khoi_tao_trong_so.png")


if __name__ == "__main__":
    fig_xor_hidden_space()
    fig_architecture()
    fig_activations()
    fig_vanishing_gradient()
    fig_backprop_graph()
    fig_boundary_epochs()
    fig_hidden_width()
    fig_learning_rate()
    fig_overfitting_dropout()
    fig_weight_init()
    print("Xong.")
