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
PALETTE = [C1, C2, C3, C4, "#7c3aed", "#0891b2", "#db2777"]


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
    frames.append(("Bước 0: khởi tạo 3 tâm, chưa gán nhãn",
                   None, centers.copy(), inertia_of(X, centers, lab)))
    for it in range(1, 3):
        lab, _ = assign(X, centers)
        frames.append((f"Vòng {it}, bước Assign: gán về tâm gần nhất",
                       lab.copy(), centers.copy(), inertia_of(X, centers, lab)))
        centers = update(X, centers, lab)
        ttl = (f"Vòng {it}, bước Update: tâm = trung bình cụm")
        frames.append((ttl, lab.copy(), centers.copy(), inertia_of(X, centers, lab)))
    # chạy tiếp tới khi hội tụ để làm panel cuối
    for _ in range(50):
        lab, _ = assign(X, centers)
        centers = update(X, centers, lab)
    frames.append(("Hội tụ: tâm và nhãn không còn đổi",
                   lab.copy(), centers.copy(), inertia_of(X, centers, lab)))

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0))
    for ax, (ttl, lb, cs, inr) in zip(axes.flat, frames):
        draw_clusters(ax, X, lb, cs, K)
        ax.set_title(ttl, fontsize=9.5)
        ax.text(.03, .04, f"inertia = {inr:.1f}", transform=ax.transAxes,
                fontsize=9.5, fontweight="bold", color="k",
                bbox=dict(fc="white", ec="0.7", alpha=.9, boxstyle="round,pad=0.3"))
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Thuật toán Lloyd: các bước Assign và Update",
                 fontweight="bold", fontsize=12)
    fig.text(.5, -.01, "Mỗi nửa bước đều làm inertia giảm; thuật toán dừng khi các tâm "
             "không còn dịch chuyển (một cực tiểu địa phương).",
             ha="center", va="top", fontsize=10, color="0.25")
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
    ax.text(.14, .62, "Inertia giảm đơn điệu và bị chặn dưới bởi 0,\n"
            "nên thuật toán chắc chắn hội tụ. Nhưng chỉ hội tụ về\n"
            "cực tiểu ĐỊA PHƯƠNG, không hứa hẹn cực tiểu toàn cục.",
            transform=ax.transAxes, fontsize=8.5, va="top",
            bbox=dict(fc="white", ec="0.75", alpha=.92, boxstyle="round,pad=0.3"))
    ax.set_ylabel("inertia $L=\\sum_k\\sum_{x\\in C_k}\\|x-\\mu_k\\|^2$")
    ax.legend(fontsize=8.5)
    ax.set_title("Inertia sau từng nửa bước của thuật toán Lloyd", fontsize=10.5)
    ax.annotate("phẳng = đã hội tụ\n(cực tiểu địa phương)", xy=(xs[-1], ys[-1]),
                xytext=(xs[-1] - 3.4, ys[-1] + (max(ys) - min(ys)) * .28),
                arrowprops=dict(arrowstyle="->", color="0.35"), fontsize=9.5, color="0.15",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=.9, ec="0.75"))
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
                        xytext=(5.4, inert[1] * .90),
                        arrowprops=dict(arrowstyle="->", color=C2), fontsize=9.5, color=C2,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=.92, ec=C2))
            ax.set_title("Elbow rõ ràng", fontsize=10)
        else:
            ax.set_title("Đường cong trơn, không có elbow", fontsize=10)
            ax.text(.32, .55, "Không có chỗ gãy!\nMọi K đều 'hợp lý như nhau'.\n"
                              "Ép chọn K ở đây là tự lừa mình:\ndữ liệu vốn không có cụm.",
                    transform=ax.transAxes, fontsize=9, color=C2,
                    bbox=dict(fc="#fff5f5", ec=C2, alpha=.9, boxstyle="round,pad=0.4"))
    fig.suptitle("Phương pháp elbow trên hai bộ dữ liệu", fontweight="bold", fontsize=11)
    fig.text(.5, -.01, "Elbow hữu ích nhưng không phải lúc nào cũng có khuỷu tay. "
             "Inertia LUÔN giảm khi K tăng (K=n thì inertia=0),\n"
             "nên không được dùng inertia thô để so sánh các giá trị K.",
             ha="center", va="top", fontsize=10, color="0.25")
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
    ax.set_title("(a) Silhouette trung bình theo K", fontsize=9.5)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + (hi - lo) * .22)
    ax.annotate(f"đỉnh tại K={best}\n(rõ hơn khuỷu tay của elbow)", xy=(best, max(sc)),
                xytext=(best + 0.8, max(sc) + (hi - lo) * .16), fontsize=8.5, color=C2,
                arrowprops=dict(arrowstyle="->", color=C2),
                bbox=dict(fc="white", ec="0.75", alpha=.92, boxstyle="round,pad=0.25"))

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
        note = ("K này TỐT: mọi cụm đều dày, vượt đường\ntrung bình, rất ít $s(i)<0$"
                if k == 4 else
                "K này XẤU: có cụm mỏng dính và nhiều\n$s(i)$ thấp hoặc âm, cụm thật bị cắt vụn")
        ax.set_ylim(0, y0 * 1.30)
        ax.text(.03, .97, note, transform=ax.transAxes, fontsize=8.5, va="top",
                color="#065f46" if k == 4 else "#7f1d1d",
                bbox=dict(fc="white", ec="0.75", alpha=.95, boxstyle="round,pad=0.3"))
        ax.set_title(f"({'b' if k == 4 else 'c'}) K={k},  $\\bar{{s}}$={s.mean():.3f}", fontsize=9.5)
    fig.suptitle(r"Chỉ số silhouette $s(i)=\dfrac{b_i-a_i}{\max(a_i,b_i)}\in[-1,1]$",
                 fontweight="bold", fontsize=10.5)
    fig.text(.5, -.02, r"$a_i$: khoảng cách trung bình tới các điểm CÙNG cụm (độ chặt);   "
             r"$b_i$: khoảng cách trung bình tới cụm KHÁC gần nhất (độ tách)",
             ha="center", va="top", fontsize=10, color="0.25")
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
    # nới đáy hai panel tán xạ để có dải trống đặt chú thích, không đè lên điểm dữ liệu
    for a in axes[:2]:
        lo, hi = a.get_ylim()
        a.set_ylim(lo - (hi - lo) * .40, hi)
    axes[0].text(.03, .03, "nghiệm TỐT", transform=axes[0].transAxes, fontsize=9.5,
                 color=C3, fontweight="bold", va="bottom",
                 bbox=dict(fc="white", ec=C3, boxstyle="round,pad=0.3"))
    axes[1].text(.03, .03, "CỰC TIỂU ĐỊA PHƯƠNG: một cụm thật bị XẺ ĐÔI,\n"
                 "hai cụm thật bị GỘP làm một\n→ inertia gấp ~6.5 lần nghiệm tốt",
                 transform=axes[1].transAxes, fontsize=8.5, color=C2, fontweight="bold", va="bottom",
                 bbox=dict(fc="white", ec=C2, alpha=.95, boxstyle="round,pad=0.3"))

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
    ax.legend(fontsize=8, loc="upper right")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 1.35)
    ax.set_title("(c) Inertia của 120 lần chạy đơn lẻ", fontsize=9.5)
    ax.text(.20, .52, "k-means++ dồn về đáy,\nrandom hay mắc kẹt", transform=ax.transAxes,
            fontsize=8.5, va="top", color="0.2",
            bbox=dict(fc="white", ec="0.75", alpha=.92, boxstyle="round,pad=0.25"))
    fig.suptitle("Ảnh hưởng của cách khởi tạo tâm tới nghiệm", fontweight="bold", fontsize=11)
    fig.text(.5, -.01, "Đây là lý do sklearn để mặc định init='k-means++' và n_init>1: "
             "chạy nhiều lần rồi giữ lại nghiệm có inertia nhỏ nhất.",
             ha="center", va="top", fontsize=10, color="0.25")
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
    fig.suptitle("Cơ chế chọn tâm khởi tạo của k-means++", fontweight="bold", fontsize=11.5)
    fig.text(.5, -.02, "Tâm tiếp theo được BỐC THĂM trong chính các điểm dữ liệu với "
             r"$p(x)=D(x)^2/\sum_{x'}D(x')^2$." "\n"
             "Nền là trường $D(x)^2$ trên mặt phẳng: điểm càng sáng thì càng xa mọi tâm đã có, càng dễ được chọn.\n"
             "Nhờ vậy các tâm khởi tạo tự trải ra khắp dữ liệu, thay vì chụm một chỗ như init='random'.",
             ha="center", va="top", fontsize=10, color="0.25")
    fig.tight_layout()
    save(fig, "06_kmeanspp_xac_suat.png")


# ---------------------------------------------------------------- 7
def fig_failure_modes():
    from sklearn.datasets import make_blobs, make_moons
    from sklearn.cluster import KMeans
    rng = np.random.default_rng(0)

    Xm, _ = make_moons(n_samples=400, noise=0.06, random_state=0)
    cases = [(Xm, 2, "Cụm cong (moons)",
              "K-Means chỉ cắt được bằng đường thẳng\n→ xẻ ngang hai trăng lưỡi liềm.\nDùng DBSCAN / Spectral.")]

    Xv = np.vstack([rng.normal([0, 0], 0.4, (200, 2)),
                    rng.normal([4, 0], 1.9, (200, 2))])
    cases.append((Xv, 2, "Phương sai rất khác nhau",
                  "Inertia phạt theo khoảng cách nên cụm rộng\nbị 'cắn' mất rìa, cụm hẹp phình ra.\nDùng GMM (covariance tự do)."))

    Xa, _ = make_blobs(n_samples=400, centers=3, cluster_std=0.6, random_state=170)
    Xa = Xa @ np.array([[0.6, -0.63], [-0.4, 0.85]])
    cases.append((Xa, 3, "Cụm kéo dài và xiên (anisotropic)",
                  "Ranh giới K-Means luôn vuông góc với đoạn\nnối 2 tâm → không ôm được cụm elip xiên.\nDùng GMM full covariance."))

    Xs = np.vstack([rng.normal([0, 0], 0.9, (600, 2)),
                    rng.normal([4.0, 0], 0.25, (25, 2)),
                    rng.normal([5.4, 0], 0.25, (25, 2))])
    cases.append((Xs, 3, "Kích thước cụm chênh lệch (600/25/25 điểm)",
                  "Inertia là TỔNG nên cụm 600 điểm đóng góp lớn hơn nhiều:\nthuật toán thà XẺ ĐÔI cụm to còn hơn tách 2 cụm nhỏ\n→ 2 cụm nhỏ thật bị GỘP làm một."))

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 9.0))
    for ax, (X, k, ttl, why) in zip(axes.flat, cases):
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        draw_clusters(ax, X, km.labels_, km.cluster_centers_, k)
        ax.set_title(ttl, fontsize=10.5, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        # nới đáy để hộp giải thích nằm trong dải trống, không đè lên điểm dữ liệu
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo - (hi - lo) * .34, hi)
        ax.text(.02, .02, "VÌ SAO HỎNG: " + why, transform=ax.transAxes, fontsize=8.5,
                color="#7f1d1d", va="bottom",
                bbox=dict(fc="#fff5f5", ec=C2, alpha=.95, boxstyle="round,pad=0.35"))
    fig.suptitle("Bốn kiểu dữ liệu khiến K-Means thất bại", fontweight="bold", fontsize=12)
    fig.text(.5, -.01, "Gốc rễ chung: K-Means giả định các cụm hình CẦU, kích thước và mật độ "
             "tương đương nhau, và gán nhãn CỨNG.",
             ha="center", va="top", fontsize=10.5, color="0.25")
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
    axes[0].set_title("(a) Sơ đồ Voronoi của các tâm", fontsize=10)
    axes[1].set_title("(b) Các cụm K-Means trên cùng dữ liệu", fontsize=10)
    axes[0].text(.98, .02, r"Biên giữa 2 tâm $\mu_i,\mu_j$ là đường TRUNG TRỰC"
                 "\n" r"$\|x-\mu_i\|=\|x-\mu_j\|$: một đường thẳng."
                 "\nMỗi ô là giao của các nửa mặt phẳng nên"
                 "\nluôn là một ĐA GIÁC LỒI.",
                 transform=axes[0].transAxes, fontsize=9, va="bottom", ha="right",
                 bbox=dict(fc="white", ec="0.6", alpha=.92, boxstyle="round,pad=0.35"))
    fig.suptitle("Ranh giới Voronoi và tính lồi của cụm K-Means",
                 fontweight="bold", fontsize=11.5)
    fig.text(.5, -.01, "Hệ quả: mọi cụm mà K-Means tìm được đều LỒI và nối liền, "
             "nên cụm hình chữ C hay hình xoắn ốc là bất khả thi.",
             ha="center", va="top", fontsize=10, color="0.25")
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
    sets = [(truth, "(a) 3 nhóm thật", None),
            (lab_raw, "(b) K-Means trên dữ liệu thô", ari(truth, lab_raw)),
            (lab_sc, "(c) K-Means sau StandardScaler", ari(truth, lab_sc))]
    for ax, (lb, ttl, a) in zip(axes, sets):
        for k in range(3):
            m = lb == k
            ax.scatter(age[m], inc[m] / 1e6, s=16, color=PALETTE[k], alpha=.8)
        ax.set_xlabel("tuổi (năm), biên độ ≈ 40")
        ax.set_ylabel("thu nhập (triệu VNĐ), biên độ ≈ 25")
        ax.set_title(ttl + (f"\nARI = {a:.2f}" if a is not None else "\n(nhãn tham chiếu)"), fontsize=10)
    # nới trần cả 3 panel (giữ cùng thang) để có dải trống đặt chú thích
    for a in axes:
        lo, hi = a.get_ylim()
        a.set_ylim(lo, hi + (hi - lo) * .48)
    axes[1].text(.02, .97, "Thu nhập tính bằng ĐỒNG (chênh hàng triệu),\n"
                           "tuổi chỉ chênh vài chục → Euclid gần như\n"
                           "CHỈ CÒN thu nhập: 2 nhóm cùng thu nhập bị\n"
                           "GỘP, nhóm thu nhập cao bị XẺ ĐÔI",
                 transform=axes[1].transAxes, fontsize=8, color="#7f1d1d", va="top",
                 bbox=dict(fc="#fff5f5", ec=C2, alpha=.95, boxstyle="round,pad=0.3"))
    fig.suptitle("Ảnh hưởng của thang đo tới kết quả K-Means",
                 fontweight="bold", fontsize=11)
    fig.text(.5, -.01, "K-Means dùng khoảng cách Euclid, nên feature có ĐƠN VỊ LỚN sẽ lấn át "
             "mọi feature khác. Hãy scale dữ liệu trước khi phân cụm.",
             ha="center", va="top", fontsize=10, color="0.25")
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
    # đặt ngoài khung để không đè lên điểm dữ liệu
    axes[2, 3].text(.99, -.02, "xám ✕ = nhiễu (nhãn −1)", transform=axes[2, 3].transAxes,
                    ha="right", va="top", fontsize=9.5, color="0.25")
    fig.suptitle("Bốn thuật toán phân cụm trên cùng ba bộ dữ liệu",
                 fontweight="bold", fontsize=11.5)
    fig.text(.5, -.005, "Chọn thuật toán chính là chọn GIẢ ĐỊNH về hình dạng cụm. "
             "K-Means và Ward thích cụm cầu; GMM ôm được cụm elip;\n"
             "DBSCAN bám theo mật độ nên xử lý được cả cụm cong.",
             ha="center", va="top", fontsize=10.5, color="0.25")
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
