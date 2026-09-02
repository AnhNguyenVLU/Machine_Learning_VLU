"""
Sinh toàn bộ hình minh hoạ cho Lab 08 - K-Means Clustering.

Chạy:  python figures.py
Kết quả: các file .png trong cùng thư mục, được nhúng vào notebook bằng markdown.
Sinh viên có thể sửa script này để tự thí nghiệm với tham số.
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
PALETTE = [C1, C2, C3, C4, "#7c3aed", "#0891b2"]


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p)
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------ tiện ích
def assign(X, centers):
    d = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return d.argmin(axis=1), d


def inertia_of(X, centers, labels):
    return float(((X - centers[labels]) ** 2).sum())


def update(X, centers, labels):
    new = centers.copy()
    for k in range(len(centers)):
        if (labels == k).any():
            new[k] = X[labels == k].mean(axis=0)
    return new


def draw_clusters(ax, X, labels, centers, K):
    if labels is None:
        ax.scatter(X[:, 0], X[:, 1], s=16, color="0.55", alpha=.8)
    else:
        for k in range(K):
            m = labels == k
            ax.scatter(X[m, 0], X[m, 1], s=16, color=PALETTE[k], alpha=.75)
    ax.scatter(centers[:, 0], centers[:, 1], marker="*", s=340,
               c=[PALETTE[k] for k in range(K)], edgecolor="k", linewidth=1.2, zorder=6)


# ---------------------------------------------------------------- 1
def fig_lloyd_steps():
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=240, centers=3, cluster_std=0.85, random_state=7)
    K = 3
    # khởi tạo cố ý lệch để nhìn rõ các tâm di chuyển
    centers = np.array([[-4.0, 8.0], [-3.0, 9.5], [-1.0, 6.5]])

    frames = []          # (tiêu đề, labels, centers, inertia)
    lab, _ = assign(X, centers)
    frames.append(("Bước 0 — KHỞI TẠO\n3 tâm đặt tuỳ tiện, chưa gán nhãn",
                   None, centers.copy(), inertia_of(X, centers, lab)))
    for it in range(1, 3):
        lab, _ = assign(X, centers)
        frames.append((f"Vòng {it} — ASSIGN\ngán mỗi điểm về tâm gần nhất",
                       lab.copy(), centers.copy(), inertia_of(X, centers, lab)))
        centers = update(X, centers, lab)
        ttl = (f"Vòng {it} — UPDATE\ntâm = trung bình các điểm trong cụm")
        frames.append((ttl, lab.copy(), centers.copy(), inertia_of(X, centers, lab)))
    # chạy tiếp tới khi hội tụ để làm panel cuối
    for _ in range(50):
        lab, _ = assign(X, centers)
        centers = update(X, centers, lab)
    frames.append(("HỘI TỤ — tâm không còn dịch chuyển\nnhãn ổn định, inertia chạm đáy ĐỊA PHƯƠNG",
                   lab.copy(), centers.copy(), inertia_of(X, centers, lab)))

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0))
    for ax, (ttl, lb, cs, inr) in zip(axes.flat, frames):
        draw_clusters(ax, X, lb, cs, K)
        ax.set_title(ttl, fontsize=9.5)
        ax.text(.03, .04, f"inertia = {inr:.1f}", transform=ax.transAxes,
                fontsize=9.5, fontweight="bold", color="k",
                bbox=dict(fc="white", ec="0.7", alpha=.9, boxstyle="round,pad=0.3"))
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Thuật toán Lloyd: lặp ASSIGN ↔ UPDATE — mỗi nửa bước đều làm inertia GIẢM",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    save(fig, "01_lloyd_tung_buoc.png")


# ---------------------------------------------------------------- 2
def fig_inertia_monotone():
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=400, centers=4, cluster_std=1.0, random_state=3)
    K = 4
    rng = np.random.default_rng(11)
    centers = X[rng.choice(len(X), K, replace=False)].astype(float)

    xs, ys, kinds = [], [], []
    t = 0.0
    for it in range(9):
        lab, _ = assign(X, centers)
        xs.append(t); ys.append(inertia_of(X, centers, lab)); kinds.append("assign")
        t += 0.5
        centers = update(X, centers, lab)
        xs.append(t); ys.append(inertia_of(X, centers, lab)); kinds.append("update")
        t += 0.5

    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.plot(xs, ys, "-", color="0.5", lw=1.4, zorder=1)
    for x, y, k in zip(xs, ys, kinds):
        ax.scatter([x], [y], s=70, zorder=3,
                   color=C1 if k == "assign" else C3,
                   marker="o" if k == "assign" else "s")
    ax.scatter([], [], color=C1, marker="o", s=70, label="sau bước ASSIGN (tối ưu theo NHÃN, giữ tâm)")
    ax.scatter([], [], color=C3, marker="s", s=70, label="sau bước UPDATE (tối ưu theo TÂM, giữ nhãn)")
    ax.set_xlabel("nửa bước của thuật toán →")
    ax.set_ylabel("inertia $L=\\sum_k\\sum_{x\\in C_k}\\|x-\\mu_k\\|^2$")
    ax.legend(fontsize=8.5)
    ax.set_title("Inertia GIẢM ĐƠN ĐIỆU và bị chặn dưới bởi 0 → thuật toán chắc chắn hội tụ\n"
                 "(hội tụ về cực tiểu ĐỊA PHƯƠNG, không hứa hẹn cực tiểu toàn cục)", fontsize=10.5)
    ax.annotate("phẳng = hội tụ", xy=(xs[-1], ys[-1]), xytext=(xs[-1] - 3.2, ys[-1] + (max(ys) - min(ys)) * .25),
                arrowprops=dict(arrowstyle="->", color="dimgray"), fontsize=9, color="dimgray")
    save(fig, "02_inertia_giam_don_dieu.png")


# ---------------------------------------------------------------- 3
def fig_elbow():
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans
    Xg, _ = make_blobs(n_samples=400, centers=4, cluster_std=0.7, random_state=42)
    rng = np.random.default_rng(0)
    Xu = rng.uniform(-6, 6, size=(400, 2))       # dữ liệu KHÔNG có cấu trúc cụm

    Ks = list(range(1, 11))
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.6))
    for col, (X, name) in enumerate([(Xg, "4 cụm tách bạch"), (Xu, "dữ liệu ĐỀU (không có cụm)")]):
        ax = axes[0, col]
        ax.scatter(X[:, 0], X[:, 1], s=14, color=C1 if col == 0 else C4, alpha=.7)
        ax.set_title(f"Dữ liệu: {name}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

        inert = [KMeans(n_clusters=k, n_init=10, random_state=0).fit(X).inertia_ for k in Ks]
        ax = axes[1, col]
        ax.plot(Ks, inert, "o-", color=C1 if col == 0 else C4, lw=2)
        ax.set_xlabel("K"); ax.set_ylabel("inertia")
        if col == 0:
            ax.axvline(4, color=C2, ls="--", lw=1.8)
            ax.annotate("KHUỶU TAY rõ ở K=4\n→ chọn K=4", xy=(4, inert[3]),
                        xytext=(5.2, inert[1] * .85),
                        arrowprops=dict(arrowstyle="->", color=C2), fontsize=9.5, color=C2)
            ax.set_title("Elbow rõ ràng", fontsize=10)
        else:
            ax.set_title("KHÔNG có khuỷu tay — đường cong trơn", fontsize=10)
            ax.text(.32, .55, "Không có chỗ gãy!\nMọi K đều 'hợp lý như nhau'.\n"
                              "Ép chọn K ở đây là tự lừa mình:\ndữ liệu vốn không có cụm.",
                    transform=ax.transAxes, fontsize=9, color=C2,
                    bbox=dict(fc="#fff5f5", ec=C2, alpha=.9, boxstyle="round,pad=0.4"))
    fig.suptitle("Elbow method: hữu ích nhưng KHÔNG phải lúc nào cũng có khuỷu tay\n"
                 "Lưu ý: inertia LUÔN giảm khi K tăng (K=n thì inertia=0) → không dùng inertia thô để so K",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    save(fig, "03_elbow_ro_va_mo_ho.png")


# ---------------------------------------------------------------- 4
def fig_silhouette():
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, silhouette_samples
    X, _ = make_blobs(n_samples=400, centers=4, cluster_std=0.75, random_state=42)

    Ks = list(range(2, 11))
    sc = [silhouette_score(X, KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X))
          for k in Ks]

    fig = plt.figure(figsize=(14, 4.4))
    ax = fig.add_subplot(1, 3, 1)
    ax.plot(Ks, sc, "o-", color=C3, lw=2)
    best = Ks[int(np.argmax(sc))]
    ax.axvline(best, color=C2, ls="--", lw=1.6)
    ax.set_xlabel("K"); ax.set_ylabel("silhouette trung bình")
    ax.set_title(f"(a) Silhouette theo K — đỉnh tại K={best}\ncó ĐỈNH rõ, dễ đọc hơn elbow", fontsize=9.5)

    for j, k in enumerate([4, 7]):
        lab = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
        s = silhouette_samples(X, lab)
        ax = fig.add_subplot(1, 3, j + 2)
        y0 = 5
        for c in range(k):
            v = np.sort(s[lab == c])
            ax.fill_betweenx(np.arange(y0, y0 + len(v)), 0, v,
                             color=PALETTE[c % len(PALETTE)], alpha=.8)
            ax.text(-0.05, y0 + len(v) / 2, str(c), va="center", ha="right", fontsize=8)
            y0 += len(v) + 8
        ax.axvline(s.mean(), color=C2, ls="--", lw=1.6)
        ax.set_xlim(-0.35, 1)
        ax.set_yticks([])
        ax.set_xlabel("$s(i)$")
        good = "TỐT" if k == 4 else "XẤU"
        note = ("mọi cụm đều dày, vượt đường trung bình,\nrất ít $s(i)<0$"
                if k == 4 else
                "có cụm mỏng dính và nhiều $s(i)$ thấp/âm\n→ K này cắt vụn cụm thật")
        ax.set_title(f"({'b' if k == 4 else 'c'}) K={k} — {good}   $\\bar{{s}}$={s.mean():.3f}\n{note}",
                     fontsize=9.5)
    fig.suptitle(r"Silhouette: $s(i)=\dfrac{b_i-a_i}{\max(a_i,b_i)}\in[-1,1]$   "
                 r"($a_i$: k/c trung bình tới CÙNG cụm — độ chặt;  "
                 r"$b_i$: k/c trung bình tới cụm KHÁC gần nhất — độ tách)",
                 fontweight="bold", fontsize=10.5)
    fig.tight_layout()
    save(fig, "04_silhouette.png")


# ---------------------------------------------------------------- 5
def fig_bad_init():
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans
    X, _ = make_blobs(n_samples=400, centers=4, cluster_std=0.7, random_state=42)

    fig = plt.figure(figsize=(14, 4.4))
    seeds = [(3, "(a) random init, seed=3"), (16, "(b) random init, seed=16")]
    for j, (sd, ttl) in enumerate(seeds):
        km = KMeans(n_clusters=4, init="random", n_init=1, random_state=sd).fit(X)
        ax = fig.add_subplot(1, 3, j + 1)
        draw_clusters(ax, X, km.labels_, km.cluster_centers_, 4)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{ttl}\ninertia = {km.inertia_:.1f}", fontsize=10)
    axes = fig.axes
    axes[0].text(.03, .03, "nghiệm TỐT", transform=axes[0].transAxes, fontsize=9.5,
                 color=C3, fontweight="bold",
                 bbox=dict(fc="white", ec=C3, boxstyle="round,pad=0.3"))
    axes[1].text(.26, .58, "CỰC TIỂU ĐỊA PHƯƠNG:\nmột cụm thật bị XẺ ĐÔI,\nhai cụm thật bị GỘP làm một\n→ inertia gấp ~6.5 lần nghiệm tốt",
                 transform=axes[1].transAxes, fontsize=8.5, color=C2, fontweight="bold", va="top",
                 bbox=dict(fc="white", ec=C2, boxstyle="round,pad=0.3"))

    ir, ip = [], []
    for sd in range(120):
        ir.append(KMeans(n_clusters=4, init="random", n_init=1, random_state=sd).fit(X).inertia_)
        ip.append(KMeans(n_clusters=4, init="k-means++", n_init=1, random_state=sd).fit(X).inertia_)
    ax = fig.add_subplot(1, 3, 3)
    bins = np.geomspace(min(ir + ip) * .9, max(ir + ip) * 1.1, 36)
    pr = 100 * np.mean(np.array(ir) < 400)
    pp = 100 * np.mean(np.array(ip) < 400)
    ax.hist(ir, bins=bins, color=C2, alpha=.6, label=f"random  ({pr:.0f}% chạm nghiệm tốt)")
    ax.hist(ip, bins=bins, color=C3, alpha=.6, label=f"k-means++  ({pp:.0f}% chạm nghiệm tốt)")
    ax.set_xscale("log")
    ax.set_xlabel("inertia sau 1 lần chạy (thang log)"); ax.set_ylabel("số lần / 120 seed")
    ax.legend(fontsize=8)
    ax.set_title("(c) 120 seed, mỗi seed chỉ chạy 1 lần\nk-means++ dồn về đáy, random hay mắc kẹt", fontsize=9.5)
    fig.suptitle("Khởi tạo quyết định nghiệm — vì sao sklearn để mặc định init='k-means++' và n_init>1",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    save(fig, "05_khoi_tao_kem.png")


# ---------------------------------------------------------------- 6
def fig_kmeanspp_prob():
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.7, random_state=42)

    chosen = [X[np.argmin(X[:, 0] + X[:, 1])]]           # tâm đầu tiên: một điểm ở góc
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    gx = np.linspace(X[:, 0].min() - 1.5, X[:, 0].max() + 1.5, 260)
    gy = np.linspace(X[:, 1].min() - 1.5, X[:, 1].max() + 1.5, 260)
    GX, GY = np.meshgrid(gx, gy)
    G = np.c_[GX.ravel(), GY.ravel()]

    for step, ax in enumerate(axes):
        C = np.array(chosen)
        D2 = ((G[:, None, :] - C[None, :, :]) ** 2).sum(2).min(1).reshape(GX.shape)
        P = D2 / D2.sum()
        ax.pcolormesh(GX, GY, P, cmap="magma", shading="auto", alpha=.85)
        d2p = ((X[:, None, :] - C[None, :, :]) ** 2).sum(2).min(1)
        d2p = d2p / d2p.sum()
        im = ax.scatter(X[:, 0], X[:, 1], s=26, c=d2p, cmap="magma",
                        edgecolor="white", linewidth=.5, zorder=4)
        ax.scatter(C[:, 0], C[:, 1], marker="*", s=380, color=C3,
                   edgecolor="white", linewidth=1.4, zorder=6)
        ax.set_title(f"Đã chọn {len(C)} tâm", fontsize=10.5)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=.046, label="$p(x)$ của mỗi ĐIỂM DỮ LIỆU")
        if step < 2:
            # chọn tâm tiếp theo = điểm dữ liệu có D^2 lớn nhất (minh hoạ; thật thì bốc thăm)
            d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(2).min(1)
            chosen.append(X[int(np.argmax(d2))])
    fig.suptitle("Cơ chế k-means++: tâm tiếp theo được BỐC THĂM trong các điểm dữ liệu với "
                 r"$p(x)=D(x)^2/\sum_{x'}D(x')^2$" "\n"
                 "(nền = trường $D(x)^2$ trên mặt phẳng; điểm càng SÁNG = càng xa mọi tâm đã có = càng dễ được chọn)\n"
                 "→ các tâm khởi tạo tự động trải ra khắp dữ liệu thay vì chụm một chỗ như init='random'",
                 fontweight="bold", fontsize=10.5)
    fig.tight_layout()
    save(fig, "06_kmeanspp_xac_suat.png")


# ---------------------------------------------------------------- 7
def fig_failure_modes():
    from sklearn.datasets import make_blobs, make_moons
    from sklearn.cluster import KMeans
    rng = np.random.default_rng(0)

    Xm, _ = make_moons(n_samples=400, noise=0.06, random_state=0)
    cases = [(Xm, 2, "Cụm CONG (moons)",
              "K-Means chỉ cắt được bằng đường thẳng\n→ xẻ ngang hai trăng lưỡi liềm.\nDùng DBSCAN / Spectral.")]

    Xv = np.vstack([rng.normal([0, 0], 0.4, (200, 2)),
                    rng.normal([4, 0], 1.9, (200, 2))])
    cases.append((Xv, 2, "Phương sai RẤT KHÁC nhau",
                  "Inertia phạt theo khoảng cách nên cụm rộng\nbị 'cắn' mất rìa, cụm hẹp phình ra.\nDùng GMM (covariance tự do)."))

    Xa, _ = make_blobs(n_samples=400, centers=3, cluster_std=0.6, random_state=170)
    Xa = Xa @ np.array([[0.6, -0.63], [-0.4, 0.85]])
    cases.append((Xa, 3, "Cụm bị KÉO DÀI / XIÊN (anisotropic)",
                  "Ranh giới K-Means luôn vuông góc với đoạn\nnối 2 tâm → không ôm được cụm elip xiên.\nDùng GMM full covariance."))

    Xs = np.vstack([rng.normal([0, 0], 0.9, (600, 2)),
                    rng.normal([4.0, 0], 0.25, (25, 2)),
                    rng.normal([5.4, 0], 0.25, (25, 2))])
    cases.append((Xs, 3, "Kích thước cụm RẤT CHÊNH LỆCH (600 / 25 / 25 điểm)",
                  "Inertia là TỔNG nên cụm 600 điểm đóng góp lớn hơn nhiều:\nthuật toán thà XẺ ĐÔI cụm to còn hơn tách 2 cụm nhỏ\n→ 2 cụm nhỏ thật bị GỘP làm một."))

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 9.0))
    for ax, (X, k, ttl, why) in zip(axes.flat, cases):
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        draw_clusters(ax, X, km.labels_, km.cluster_centers_, k)
        ax.set_title(ttl, fontsize=10.5, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(.02, .02, "VÌ SAO HỎNG: " + why, transform=ax.transAxes, fontsize=8.5,
                color="#7f1d1d", va="bottom",
                bbox=dict(fc="#fff5f5", ec=C2, alpha=.92, boxstyle="round,pad=0.35"))
    fig.suptitle("Bốn kiểu dữ liệu khiến K-Means thất bại\n"
                 "Gốc rễ: K-Means giả định cụm CẦU, kích thước và mật độ tương đương, và gán CỨNG",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    save(fig, "07_kmeans_that_bai.png")


# ---------------------------------------------------------------- 8
def fig_voronoi():
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans
    X, _ = make_blobs(n_samples=350, centers=5, cluster_std=0.9, random_state=5)
    km = KMeans(n_clusters=5, n_init=10, random_state=0).fit(X)
    C = km.cluster_centers_

    pad = 2.0
    gx = np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, 500)
    gy = np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, 500)
    GX, GY = np.meshgrid(gx, gy)
    Z = ((np.c_[GX.ravel(), GY.ravel()][:, None, :] - C[None]) ** 2).sum(2).argmin(1).reshape(GX.shape)

    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([c for c in PALETTE[:5]])

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0))
    for ax, showpts in zip(axes, [False, True]):
        ax.pcolormesh(GX, GY, Z, cmap=cmap, alpha=.20, shading="auto")
        ax.contour(GX, GY, Z, levels=np.arange(0.5, 5, 1), colors="k", linewidths=1.2)
        if showpts:
            for k in range(5):
                m = km.labels_ == k
                ax.scatter(X[m, 0], X[m, 1], s=14, color=PALETTE[k], alpha=.85)
        ax.scatter(C[:, 0], C[:, 1], marker="*", s=340,
                   c=[PALETTE[k] for k in range(5)], edgecolor="k", linewidth=1.2, zorder=6)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    axes[0].set_title("Ranh giới do K-Means sinh ra = sơ đồ VORONOI của các tâm\n"
                      "mỗi ô là giao của các nửa mặt phẳng → ĐA GIÁC LỒI", fontsize=10)
    axes[1].set_title("Hệ quả: mọi cụm K-Means tìm được đều LỒI và nối liền\n"
                      "→ cụm hình chữ C, hình xoắn ốc là bất khả thi", fontsize=10)
    axes[0].text(.02, .02, r"biên giữa 2 tâm $\mu_i,\mu_j$ là đường TRUNG TRỰC:"
                 "\n" r"$\|x-\mu_i\|=\|x-\mu_j\|$ — một đường thẳng",
                 transform=axes[0].transAxes, fontsize=9,
                 bbox=dict(fc="white", ec="0.6", alpha=.92, boxstyle="round,pad=0.35"))
    fig.suptitle("Vì sao K-Means chỉ tạo được cụm lồi", fontweight="bold", fontsize=11.5)
    fig.tight_layout()
    save(fig, "08_voronoi_cum_loi.png")


# ---------------------------------------------------------------- 9
def fig_need_scaling():
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(1)
    n = 150
    # 3 nhóm khách hàng: tuổi (năm) và thu nhập (VNĐ/tháng)
    # A: trẻ - thu nhập thấp;  B: trung niên - thu nhập thấp (CÙNG thu nhập với A!);  C: thu nhập cao
    age = np.r_[rng.normal(24, 2.5, n), rng.normal(52, 2.5, n), rng.normal(38, 2.5, n)]
    inc = np.r_[rng.normal(12e6, 1.5e6, n), rng.normal(12e6, 1.5e6, n), rng.normal(30e6, 4.0e6, n)]
    truth = np.r_[np.zeros(n), np.ones(n), 2 * np.ones(n)].astype(int)
    X = np.c_[age, inc]

    lab_raw = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(X)
    Xs = StandardScaler().fit_transform(X)
    lab_sc = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(Xs)

    from sklearn.metrics import adjusted_rand_score as ari
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
    sets = [(truth, "(a) 3 nhóm THẬT", None),
            (lab_raw, "(b) K-Means trên dữ liệu THÔ", ari(truth, lab_raw)),
            (lab_sc, "(c) K-Means sau StandardScaler", ari(truth, lab_sc))]
    for ax, (lb, ttl, a) in zip(axes, sets):
        for k in range(3):
            m = lb == k
            ax.scatter(age[m], inc[m] / 1e6, s=16, color=PALETTE[k], alpha=.8)
        ax.set_xlabel("tuổi (năm)  —  biên độ ≈ 40")
        ax.set_ylabel("thu nhập (triệu VNĐ)  —  biên độ ≈ 25")
        ax.set_title(ttl + (f"\nARI = {a:.2f}" if a is not None else "\n(nhãn tham chiếu)"), fontsize=10)
    axes[1].text(.02, .50, "Thu nhập tính bằng ĐỒNG\n(chênh hàng triệu), tuổi\nchỉ chênh vài chục →\n"
                           "Euclid gần như CHỈ CÒN\nthu nhập: 2 nhóm cùng\nthu nhập bị GỘP, nhóm\n"
                           "thu nhập cao bị XẺ ĐÔI",
                 transform=axes[1].transAxes, fontsize=8, color="#7f1d1d", va="top",
                 bbox=dict(fc="#fff5f5", ec=C2, alpha=.92, boxstyle="round,pad=0.3"))
    fig.suptitle("K-Means dùng khoảng cách Euclid → feature có ĐƠN VỊ LỚN sẽ lấn át. LUÔN scale trước!",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    save(fig, "09_vi_sao_phai_scale.png")


# ---------------------------------------------------------------- 10
def fig_compare_algos():
    from sklearn.datasets import make_blobs, make_moons, make_circles
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(0)

    Xb, _ = make_blobs(n_samples=400, centers=3, cluster_std=0.7, random_state=170)
    Xa = StandardScaler().fit_transform(Xb @ np.array([[0.6, -0.63], [-0.4, 0.85]]))
    Xm = StandardScaler().fit_transform(make_moons(n_samples=400, noise=0.06, random_state=0)[0])
    Xc = StandardScaler().fit_transform(make_circles(n_samples=400, factor=.45, noise=.05, random_state=0)[0])

    rows = [(Xa, 3, "cụm elip xiên"), (Xm, 2, "hai trăng"), (Xc, 2, "vòng lồng nhau")]
    algos = ["K-Means", "GMM (full cov.)", "Agglomerative (ward)", "DBSCAN"]

    fig, axes = plt.subplots(3, 4, figsize=(14.5, 10.5))
    for i, (X, k, name) in enumerate(rows):
        preds = [
            KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X),
            GaussianMixture(n_components=k, covariance_type="full", random_state=0).fit_predict(X),
            AgglomerativeClustering(n_clusters=k).fit_predict(X),
            DBSCAN(eps=0.3, min_samples=6).fit_predict(X),
        ]
        for j, (p, alg) in enumerate(zip(preds, algos)):
            ax = axes[i, j]
            for c in np.unique(p):
                m = p == c
                col = "0.6" if c == -1 else PALETTE[int(c) % len(PALETTE)]
                ax.scatter(X[m, 0], X[m, 1], s=11,
                           color=col, alpha=.85, marker="x" if c == -1 else "o")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(alg, fontsize=11, fontweight="bold")
            if j == 0:
                ax.set_ylabel(name, fontsize=10.5)
    axes[2, 3].text(.03, .07, "xám ✕ = nhiễu (nhãn −1)", transform=axes[2, 3].transAxes,
                    fontsize=8.5, color="0.35")
    fig.suptitle("Cùng dữ liệu, bốn thuật toán: chọn thuật toán = chọn GIẢ ĐỊNH về hình dạng cụm\n"
                 "K-Means/Ward thích cụm cầu; GMM ôm được elip; DBSCAN bám mật độ nên xử lý được cụm cong",
                 fontweight="bold", fontsize=11.5)
    fig.tight_layout()
    save(fig, "10_so_sanh_thuat_toan.png")


if __name__ == "__main__":
    fig_lloyd_steps()
    fig_inertia_monotone()
    fig_elbow()
    fig_silhouette()
    fig_bad_init()
    fig_kmeanspp_prob()
    fig_failure_modes()
    fig_voronoi()
    fig_need_scaling()
    fig_compare_algos()
    print("Xong.")
