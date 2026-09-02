"""
Sinh toàn bộ hình minh hoạ cho Lab 07 - Đánh giá mô hình phân loại.

Chạy:  python figures.py
Kết quả: các file .png trong cùng thư mục, được nhúng vào notebook bằng markdown.
Sinh viên có thể sửa script này để tự thí nghiệm (đổi ngưỡng, đổi tỷ lệ mất cân bằng...).
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

# Bộ điểm số dùng chung cho hình 02 và 03 để hai hình khớp nhau tuyệt đối
RNG = np.random.default_rng(7)
NEG = RNG.normal(0.32, 0.135, 4000)     # điểm số model chấm cho lớp ÂM
POS = RNG.normal(0.63, 0.135, 4000)     # điểm số model chấm cho lớp DƯƠNG
THRESHOLDS = [0.35, 0.48, 0.62]


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p)
    plt.close(fig)
    print("wrote", name)


def _pdf(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))


def _counts(t):
    """TP, FP, TN, FN tại ngưỡng t (dự đoán dương nếu score >= t)."""
    TP = int((POS >= t).sum()); FN = int((POS < t).sum())
    FP = int((NEG >= t).sum()); TN = int((NEG < t).sum())
    return TP, FP, TN, FN


# ---------------------------------------------------------------- 1
def fig_confusion_anatomy():
    """Giải phẫu confusion matrix theo đúng layout của sklearn."""
    cm = np.array([[85, 15], [10, 90]])          # [[TN, FP], [FN, TP]]
    names = [["TN (True Negative)", "FP (False Positive)"],
             ["FN (False Negative)", "TP (True Positive)"]]
    plain = [["model nói KHÔNG\nvà đúng là KHÔNG\n(đoán đúng, thầm lặng)",
              "model nói CÓ\nnhưng thật ra KHÔNG\n(BÁO ĐỘNG GIẢ)"],
             ["model nói KHÔNG\nnhưng thật ra CÓ\n(BỎ SÓT, nguy hiểm nhất\ntrong y tế)",
              "model nói CÓ\nvà đúng là CÓ\n(bắt trúng)"]]
    cols = ["#dcfce7", "#fee2e2", "#fee2e2", "#dcfce7"]

    fig, ax = plt.subplots(figsize=(9.2, 6.6))
    ax.set_xlim(0, 2); ax.set_ylim(0, 2); ax.invert_yaxis()
    ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor=cols[i * 2 + j],
                                   edgecolor="k", linewidth=1.6))
            ax.text(j + .5, i + .18, names[i][j], ha="center", va="center",
                    fontsize=11.5, fontweight="bold")
            ax.text(j + .5, i + .44, f"= {cm[i, j]} mẫu", ha="center", va="center",
                    fontsize=10.5, color="#555555")
            ax.text(j + .5, i + .74, plain[i][j], ha="center", va="center",
                    fontsize=9.2)
    ax.set_xticks([.5, 1.5]); ax.set_yticks([.5, 1.5])
    ax.set_xticklabels(["Dự đoán: NEGATIVE (0)", "Dự đoán: POSITIVE (1)"],
                       fontsize=10.5, fontweight="bold")
    ax.set_yticklabels(["Thật: NEGATIVE (0)", "Thật: POSITIVE (1)"],
                       fontsize=10.5, fontweight="bold")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Confusion matrix theo layout của sklearn",
                 fontweight="bold", fontsize=12, pad=16)
    ax.text(1.0, 2.30, "Mẹo đọc tên: chữ thứ hai là ĐIỀU MODEL NÓI, "
                       "chữ thứ nhất là MODEL NÓI ĐÚNG HAY SAI.\n"
                       "\"False Negative\" = model nói Negative, và nó SAI.",
            ha="center", fontsize=9.5, color="#444444",
            bbox=dict(boxstyle="round,pad=.45", fc="#f8f8f8", ec="lightgray"))
    ax.text(1.0, 2.58, "cm = confusion_matrix(y_true, y_pred)   →   "
                       "TN=cm[0,0]  FP=cm[0,1]  FN=cm[1,0]  TP=cm[1,1]",
            ha="center", fontsize=9.5, family="monospace", color=C1)
    save(fig, "01_giai_phau_confusion_matrix.png")


# ---------------------------------------------------------------- 2
def fig_two_distributions():
    """Hình neo của cả bài: hai phân phối điểm số + một ngưỡng cắt."""
    x = np.linspace(0, 1, 600)
    pn, pp = _pdf(x, .32, .135), _pdf(x, .63, .135)

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.9), sharey=True)
    tags = ["Ngưỡng thấp 0.35 (dễ dãi)",
            "Ngưỡng vừa 0.48",
            "Ngưỡng cao 0.62 (khắt khe)"]
    notes = ["bắt được gần hết ca dương (recall cao)\nnhưng báo động giả rất nhiều (precision thấp)",
             "cân bằng giữa bỏ sót và báo động giả",
             "hầu như không báo động giả (precision cao)\nnhưng bỏ sót rất nhiều (recall thấp)"]
    for ax, t, tag, note in zip(axes, THRESHOLDS, tags, notes):
        L, R = x < t, x >= t
        ax.fill_between(x, 0, pn, where=L, color=C1, alpha=.45)
        ax.fill_between(x, 0, pn, where=R, color=C4, alpha=.75)
        ax.fill_between(x, 0, pp, where=L, color="#7c3aed", alpha=.75)
        ax.fill_between(x, 0, pp, where=R, color=C2, alpha=.45)
        ax.plot(x, pn, color=C1, lw=1.8)
        ax.plot(x, pp, color=C2, lw=1.8)
        ax.axvline(t, color="k", lw=2.4)

        TP, FP, TN, FN = _counts(t)
        prec = TP / (TP + FP); rec = TP / (TP + FN)
        ax.text(t + .015, 3.62, f"ngưỡng = {t}", fontsize=9.5, fontweight="bold")

        def tag_at(xx, yy, txt, col):
            ax.text(xx, yy, txt, color=col, fontsize=12, fontweight="bold",
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=.22", fc="white", ec=col, alpha=.93))
        tag_at(.13, 2.55, "TN", C1)
        tag_at(min(t + .13, .87), 2.55, "TP", C2)
        tag_at(max(t - .085, .05), .42, "FN", "#7c3aed")
        tag_at(min(t + .085, .95), .42, "FP", C4)
        ax.set_title(tag + f"\nprecision = {prec:.2f} | recall = {rec:.2f}", fontsize=10)
        ax.set_xlabel("điểm số model chấm cho mẫu (score)")
        ax.set_xlim(0, 1); ax.set_ylim(0, 4.4)
        ax.text(.5, -.31, note, transform=ax.transAxes, ha="center", va="top",
                fontsize=9, color="#444444")
    axes[0].set_ylabel("mật độ mẫu")
    axes[0].plot([], [], color=C1, lw=6, alpha=.6, label="lớp ÂM thật sự")
    axes[0].plot([], [], color=C2, lw=6, alpha=.6, label="lớp DƯƠNG thật sự")
    axes[0].legend(fontsize=8.5, loc="upper left", framealpha=.95)
    fig.suptitle("Hai phân phối điểm số và một ngưỡng cắt",
                 fontweight="bold", fontsize=11.5)
    fig.tight_layout()
    save(fig, "02_hai_phan_phoi_va_nguong.png")


# ---------------------------------------------------------------- 3
def fig_threshold_to_roc():
    """ROC sinh ra từ đâu: quét ngưỡng trên chính hình 02."""
    from sklearn.metrics import roc_curve, roc_auc_score
    yy = np.r_[np.zeros(len(NEG)), np.ones(len(POS))]
    ss = np.r_[NEG, POS]
    fpr, tpr, thr = roc_curve(yy, ss)
    auc = roc_auc_score(yy, ss)

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.4))

    # trái: nhắc lại hai phân phối + 3 ngưỡng
    x = np.linspace(0, 1, 600)
    ax = axes[0]
    ax.fill_between(x, 0, _pdf(x, .32, .135), color=C1, alpha=.35, label="lớp ÂM")
    ax.fill_between(x, 0, _pdf(x, .63, .135), color=C2, alpha=.35, label="lớp DƯƠNG")
    for t, mk in zip(THRESHOLDS, ["A", "B", "C"]):
        ax.axvline(t, color="k", lw=1.8, ls="--")
        ax.text(t, 3.35, f" {mk}\n t={t}", fontsize=9.5, fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 4.0)
    ax.set_xlabel("điểm số"); ax.set_ylabel("mật độ")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_title("Bước 1: trượt ngưỡng", fontsize=10.5)

    # phải: ROC + 3 điểm tương ứng
    ax = axes[1]
    ax.plot(fpr, tpr, color=C1, lw=2.6, label=f"ROC (AUC = {auc:.3f})")
    ax.fill_between(fpr, 0, tpr, color=C1, alpha=.12)
    ax.plot([0, 1], [0, 1], "k--", lw=1.4, label="đoán mò (AUC = 0.5)")
    rows = []
    for t, mk in zip(THRESHOLDS, ["A", "B", "C"]):
        TP, FP, TN, FN = _counts(t)
        f, r = FP / (FP + TN), TP / (TP + FN)
        ax.scatter([f], [r], s=110, color=C2, zorder=5, edgecolor="k", linewidth=.8)
        # chỉ dán nhãn NGẮN cạnh điểm; số liệu gom vào một bảng ở vùng trống
        ax.text(f + .035, r - .055, mk, color=C2, fontsize=11.5, fontweight="bold",
                va="center", ha="left", zorder=6,
                bbox=dict(boxstyle="round,pad=.15", fc="white", ec="none", alpha=.85))
        rows.append(f"{mk} (t={t}):  FPR={f:.2f},  TPR={r:.2f}")
    ax.text(.52, .42, "\n".join(rows), fontsize=9, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=.35", fc="white", ec="lightgray"))
    ax.scatter([0], [1], marker="*", s=280, color=C3, zorder=6, edgecolor="k", linewidth=.6)
    ax.text(.045, 1.02, "model hoàn hảo", color=C3, fontsize=9, fontweight="bold",
            va="center", ha="left",
            bbox=dict(boxstyle="round,pad=.2", fc="white", ec="none", alpha=.9))
    ax.set_xlabel("FPR = FP/(FP+TN) = 1 − Specificity")
    ax.set_ylabel("TPR = TP/(TP+FN) = Recall")
    ax.set_xlim(-.02, 1.02); ax.set_ylim(-.02, 1.09)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title("Bước 2: mỗi ngưỡng là một điểm ROC", fontsize=10.5)

    fig.suptitle("Từ ngưỡng tới đường ROC", fontweight="bold", fontsize=11.5)
    fig.tight_layout()
    save(fig, "03_tu_nguong_den_roc.png")


# ---------------------------------------------------------------- 4
def fig_pr_tradeoff():
    """Precision, Recall, F1 theo ngưỡng, và vì sao 0.5 không thiêng liêng."""
    rng = np.random.default_rng(11)
    neg = rng.normal(.35, .15, 5000)      # dữ liệu hơi mất cân bằng: ~11% lớp dương
    pos = rng.normal(.62, .15, 600)

    ts = np.linspace(0.02, 0.98, 400)
    P, R, F = [], [], []
    for t in ts:
        TP = (pos >= t).sum(); FN = (pos < t).sum(); FP = (neg >= t).sum()
        p = TP / max(TP + FP, 1); r = TP / max(TP + FN, 1)
        P.append(p); R.append(r)
        F.append(2 * p * r / (p + r) if p + r > 0 else 0)
    P, R, F = map(np.array, (P, R, F))
    best = int(np.argmax(F))
    i5 = int(np.argmin(np.abs(ts - .5)))

    fig, ax = plt.subplots(figsize=(9.8, 5.6))
    ax.plot(ts, P, color=C1, lw=2.4, label="Precision: 'báo động có đáng tin?'")
    ax.plot(ts, R, color=C2, lw=2.4, label="Recall: 'có bỏ sót ca nào không?'")
    ax.plot(ts, F, color=C3, lw=2.8, label="F1 (trung bình điều hoà)")
    # đường dóng chỉ vẽ trong vùng có đường cong, không chạy vào dải chú thích
    ax.plot([ts[best]] * 2, [0, 1.02], color=C3, ls="--", lw=1.6)
    ax.scatter([ts[best]], [F[best]], s=120, color=C3, zorder=5, edgecolor="k", linewidth=.7)
    ax.annotate(f"F1 LỚN NHẤT = {F[best]:.3f}\ntại ngưỡng {ts[best]:.2f}",
                xy=(ts[best], F[best]), xytext=(.60, 1.31),
                fontsize=9.5, va="top", ha="left",
                arrowprops=dict(arrowstyle="->", color=C3),
                bbox=dict(boxstyle="round,pad=.35", fc="#ecfdf5", ec=C3))
    ax.plot([.5, .5], [0, 1.02], color="gray", ls=":", lw=1.8)
    ax.scatter([.5], [F[i5]], s=100, color="gray", zorder=5, edgecolor="k", linewidth=.7)
    ax.annotate(f"ngưỡng mặc định 0.5\nF1 chỉ = {F[i5]:.3f}\n(kém hơn {100*(F[best]-F[i5]):.1f} điểm)",
                xy=(.5, F[i5]), xytext=(.025, .045), fontsize=9.5,
                va="top", ha="left",
                arrowprops=dict(arrowstyle="->", color="gray"),
                bbox=dict(boxstyle="round,pad=.35", fc="#f8f8f8", ec="lightgray"))
    ax.set_xlabel("ngưỡng quyết định"); ax.set_ylabel("giá trị chỉ số")
    ax.set_ylim(-.17, 1.36)
    ax.legend(fontsize=9, loc="upper left", framealpha=.95)
    ax.set_title("Precision, recall và F1 theo ngưỡng\n"
                 "(dữ liệu ~11% lớp dương)",
                 fontweight="bold", fontsize=11)
    ax.text(.985, .015, "Chọn ngưỡng là một QUYẾT ĐỊNH KINH DOANH,\nkhông phải mặc định của thư viện",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color=C2,
            bbox=dict(boxstyle="round,pad=.35", fc="#fef2f2", ec=C2))
    save(fig, "04_precision_recall_theo_nguong.png")


# ---------------------------------------------------------------- 5
def fig_roc_vs_pr():
    """Khi mất cân bằng: ROC vẫn 'đẹp' trong khi PR sụp đổ."""
    from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
    rng = np.random.default_rng(1)

    def make(n_pos, n_neg):
        s = np.r_[rng.normal(.32, .135, n_neg), rng.normal(.63, .135, n_pos)]
        y = np.r_[np.zeros(n_neg), np.ones(n_pos)]
        return y, s

    sets = [("Cân bằng 50/50\n(5000 dương / 5000 âm)", make(5000, 5000), C1),
            ("Mất cân bằng 1/99\n(100 dương / 9900 âm)", make(100, 9900), C2)]

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2))
    for name, (y, s), col in sets:
        fpr, tpr, _ = roc_curve(y, s)
        axes[0].plot(fpr, tpr, lw=2.4, color=col,
                     label=f"{name.splitlines()[0]}, AUC = {roc_auc_score(y, s):.3f}")
        p, r, _ = precision_recall_curve(y, s)
        ap = average_precision_score(y, s)
        axes[1].plot(r, p, lw=2.4, color=col,
                     label=f"{name.splitlines()[0]}, AP = {ap:.3f}")
        axes[1].axhline(y.mean(), color=col, ls=":", lw=1.6)
        # baseline cao thì dán nhãn sát mép PHẢI, baseline thấp thì mép TRÁI
        base = y.mean()
        ha, xt = ("right", .985) if base > .2 else ("left", .015)
        axes[1].text(xt, base + .035, f"baseline = tỷ lệ lớp dương = {base:.2f}",
                     color=col, fontsize=8.8, ha=ha, va="bottom",
                     bbox=dict(boxstyle="round,pad=.2", fc="white", ec="none", alpha=.9))

    axes[0].plot([0, 1], [0, 1], "k--", lw=1.2)
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR (Recall)")
    axes[0].legend(fontsize=8.8, loc="lower right")
    axes[0].set_title("Đường ROC ở hai mức mất cân bằng", fontsize=10.5)
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_ylim(0, 1.24)
    axes[1].legend(fontsize=8.8, loc="upper right", framealpha=.95)
    axes[1].set_title("Đường Precision-Recall tương ứng", fontsize=10.5)

    fig.suptitle("ROC và PR khi dữ liệu mất cân bằng",
                 fontweight="bold", fontsize=11.5)
    fig.tight_layout()
    save(fig, "05_roc_vs_pr_khi_mat_can_bang.png")


# ---------------------------------------------------------------- 6
def fig_accuracy_lies():
    """Model 'luôn đoán lớp đa số' trên dữ liệu 95/5 nhìn qua 5 chỉ số."""
    from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                                 balanced_accuracy_score, cohen_kappa_score)
    y = np.r_[np.zeros(950), np.ones(50)]
    dummy = np.zeros(1000)
    rng = np.random.default_rng(0)
    # một model "thật sự có học": bắt được 60% ca hiếm, sai 3% ca thường
    real = y.copy()
    real[(y == 1) & (rng.random(1000) < .40)] = 0
    real[(y == 0) & (rng.random(1000) < .03)] = 1

    metrics = [("Accuracy", accuracy_score),
               ("Balanced\naccuracy", balanced_accuracy_score),
               ("F1 (lớp hiếm)", lambda a, b: f1_score(a, b, zero_division=0)),
               ("Cohen's\nKappa", cohen_kappa_score),
               ("MCC", matthews_corrcoef)]
    v_dummy = [f(y, dummy) for _, f in metrics]
    v_real = [f(y, real) for _, f in metrics]
    names = [n for n, _ in metrics]

    fig, ax = plt.subplots(figsize=(10.4, 5.2))
    xpos = np.arange(len(names))
    b1 = ax.bar(xpos - .2, v_dummy, .4, color=C2, label='Model "ngốc": luôn đoán lớp 0')
    b2 = ax.bar(xpos + .2, v_real, .4, color=C3, label='Model thật sự có học')
    for bars, vals in [(b1, v_dummy), (b2, v_real)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + .02, f"{v:.3f}",
                    ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(xpos); ax.set_xticklabels(names, fontsize=9.5)
    ax.axhline(0, color="k", lw=1)
    ax.set_ylim(-.08, 1.20); ax.set_ylabel("giá trị chỉ số")
    ax.legend(fontsize=9.5, loc="upper right", framealpha=.95)
    # hai hộp chú thích đặt ở khoảng trống phía trên các cột thấp (F1/Kappa/MCC)
    ax.annotate("Model NGỐC có accuracy CAO HƠN model tốt\n"
                "(0.950 > 0.943), dù nó bỏ sót 100% ca hiếm.",
                xy=(-.2, .96), xytext=(1.55, .99), fontsize=9.5, color=C2,
                va="top", ha="left",
                arrowprops=dict(arrowstyle="->", color=C2),
                bbox=dict(boxstyle="round,pad=.35", fc="#fef2f2", ec=C2))
    ax.annotate("F1 / Kappa / MCC = 0 → lột mặt nạ ngay.\n"
                "Balanced accuracy = 0.5 = đúng bằng tung đồng xu.",
                xy=(1.8, .045), xytext=(1.55, .82), fontsize=9.5, color="#444444",
                va="top", ha="left",
                arrowprops=dict(arrowstyle="->", color="dimgray"),
                bbox=dict(boxstyle="round,pad=.35", fc="#f8f8f8", ec="lightgray"))
    ax.set_title("Năm chỉ số trên dữ liệu 95/5",
                 fontweight="bold", fontsize=11.5)
    save(fig, "06_accuracy_lua_doi.png")


# ---------------------------------------------------------------- 7
def fig_macro_micro_weighted():
    """Ba cách gộp chỉ số đa lớp trên một ví dụ 3 lớp mất cân bằng."""
    # confusion matrix 3 lớp: hàng = thật, cột = dự đoán
    cm = np.array([[880,  15,   5],
                   [ 30,  55,  15],
                   [ 12,   6,  12]])
    labels = ["Lớp A\n(900 mẫu)", "Lớp B\n(100 mẫu)", "Lớp C\n(30 mẫu)"]
    TP = np.diag(cm).astype(float)
    FP = cm.sum(0) - TP
    FN = cm.sum(1) - TP
    prec = TP / (TP + FP); rec = TP / (TP + FN)
    f1 = 2 * prec * rec / (prec + rec)
    sup = cm.sum(1)

    macro = f1.mean()
    weighted = (f1 * sup).sum() / sup.sum()
    micro = TP.sum() / (TP.sum() + FP.sum())        # = accuracy

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.2),
                             gridspec_kw={"width_ratios": [1.15, 1]})

    ax = axes[0]
    xpos = np.arange(3)
    ax.bar(xpos - .26, prec, .25, color=C1, label="Precision")
    ax.bar(xpos, rec, .25, color=C2, label="Recall")
    ax.bar(xpos + .26, f1, .25, color=C3, label="F1")
    for i in range(3):
        ax.text(i + .26, f1[i] + .02, f"{f1[i]:.2f}", ha="center", fontsize=9,
                fontweight="bold", color=C3)
    ax.set_xticks(xpos); ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylim(0, 1.12); ax.set_ylabel("giá trị")
    ax.legend(fontsize=9, loc="upper right", framealpha=.95)
    ax.set_title("Chỉ số của từng lớp", fontsize=10.5)

    ax = axes[1]
    vals = [macro, weighted, micro]
    names = ["MACRO\n(trung bình cộng\nF1 của 3 lớp)",
             "WEIGHTED\n(trung bình có trọng số\ntheo số mẫu)",
             "MICRO\n(gộp hết TP/FP/FN\nrồi tính 1 lần = accuracy)"]
    bars = ax.bar(range(3), vals, .55, color=[C4, C1, "#7c3aed"])
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + .015, f"{v:.3f}",
                ha="center", fontsize=12, fontweight="bold")
    ax.set_xticks(range(3)); ax.set_xticklabels(names, fontsize=8.8)
    ax.set_ylim(0, 1.12); ax.set_ylabel("F1 tổng hợp")
    ax.set_title("Ba cách gộp F1 đa lớp", fontsize=10.5)
    ax.annotate("Chênh nhau\n%.0f điểm phần trăm!" % ((micro - macro) * 100),
                xy=(.10, .71), xytext=(0, .99), ha="center", va="top",
                fontsize=10, color=C2, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C2),
                bbox=dict(boxstyle="round,pad=.35", fc="#fef2f2", ec=C2))

    fig.suptitle("Macro, weighted và micro trên cùng một model",
                 fontweight="bold", fontsize=11.5)
    fig.tight_layout()
    save(fig, "07_macro_micro_weighted.png")


# ---------------------------------------------------------------- 8
def fig_calibration():
    """Reliability diagram: xác suất model nói có khớp tần suất thực tế không?"""
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss
    rng = np.random.default_rng(3)
    n = 20000
    p_true = rng.beta(2, 2, n)                 # xác suất thật của từng mẫu
    y = (rng.random(n) < p_true).astype(int)

    def warp(p, a):
        lg = np.log(p / (1 - p))
        return 1 / (1 + np.exp(-a * lg))

    models = [("Quá TỰ TIN (over-confident)\nđẩy xác suất về 0 và 1", warp(p_true, 2.6), C2),
              ("ĐÃ HIỆU CHỈNH tốt (calibrated)", p_true, C3),
              ("THIẾU tự tin (under-confident)\nco cụm quanh 0.5", warp(p_true, 0.42), C1)]

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.3),
                             gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", lw=1.6, label="hiệu chỉnh hoàn hảo")
    for name, p, col in models:
        fr, mp = calibration_curve(y, p, n_bins=12, strategy="quantile")
        bs = brier_score_loss(y, p)
        ax.plot(mp, fr, "o-", color=col, lw=2.2, ms=5,
                label=f"{name.splitlines()[0]}  (Brier = {bs:.4f})")
    ax.set_xlabel("xác suất model dự đoán (trung bình mỗi nhóm)")
    ax.set_ylabel("tần suất dương THỰC TẾ trong nhóm")
    ax.legend(fontsize=8.6, loc="upper left")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Đường hiệu chỉnh (reliability diagram)", fontsize=10.5)
    ax.text(.45, .10, "Ở vùng xác suất cao, đường đỏ nằm\nDƯỚI đường chéo: model nói 90%\n"
            "nhưng thực tế chỉ ~72% đúng → NÓI QUÁ",
            fontsize=8.8, color=C2,
            bbox=dict(boxstyle="round,pad=.35", fc="#fef2f2", ec=C2))

    ax = axes[1]
    for name, p, col in models:
        ax.hist(p, bins=40, histtype="step", lw=2.2, color=col, density=True,
                label=name.splitlines()[0])
    ax.set_xlabel("xác suất model dự đoán")
    ax.set_ylabel("mật độ")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 1.28)          # chừa chỗ cho legend, tránh đè đỉnh cột
    ax.legend(fontsize=8.6, loc="upper right", framealpha=.95)
    ax.set_title("Phân bố xác suất đầu ra", fontsize=10.5)

    fig.suptitle("Hiệu chỉnh xác suất và Brier score",
                 fontweight="bold", fontsize=11.5)
    fig.tight_layout()
    save(fig, "08_calibration_curve.png")


# ---------------------------------------------------------------- 9
def fig_cross_validation():
    """Sơ đồ k-fold và stratified k-fold."""
    k = 5
    fig, axes = plt.subplots(3, 1, figsize=(11.6, 7.6),
                             gridspec_kw={"height_ratios": [1, 3.1, 1.5]})

    # (a) một lần train_test_split
    ax = axes[0]
    ax.add_patch(Rectangle((0, 0), .8, .6, color=C1, alpha=.55))
    ax.add_patch(Rectangle((.8, 0), .2, .6, color=C4, alpha=.9))
    ax.text(.4, .3, "TRAIN (80%)", ha="center", va="center", fontsize=11, color="white",
            fontweight="bold")
    ax.text(.9, .3, "TEST", ha="center", va="center", fontsize=10, color="white",
            fontweight="bold")
    ax.text(1.02, .3, "  ← chỉ MỘT con số, phụ thuộc hoàn toàn vào việc\n"
                      "      random_state may hay rủi", va="center", fontsize=9.5, color=C2)
    ax.set_xlim(0, 1.75); ax.set_ylim(-.1, .7); ax.axis("off")
    ax.set_title("Cách 1: train_test_split một lần",
                 fontsize=11, fontweight="bold", loc="left")

    # (b) k-fold
    ax = axes[1]
    for i in range(k):
        yb = k - 1 - i
        for j in range(k):
            is_val = (j == i)
            ax.add_patch(Rectangle((j * .2, yb), .195, .78,
                                   color=C4 if is_val else C1,
                                   alpha=.9 if is_val else .5))
            ax.text(j * .2 + .0975, yb + .39, "VAL" if is_val else "train",
                    ha="center", va="center", fontsize=9,
                    color="white", fontweight="bold" if is_val else "normal")
        ax.text(-.015, yb + .39, f"Vòng {i+1}", ha="right", va="center", fontsize=9.5)
        ax.text(1.02, yb + .39, f"→ score$_{i+1}$", va="center", fontsize=9.5, color="dimgray")
    ax.set_xlim(-.16, 1.35); ax.set_ylim(-.62, k + .05); ax.axis("off")
    ax.set_title("Cách 2: k-fold cross-validation (k = 5)",
                 fontsize=11, fontweight="bold", loc="left")
    ax.text(.5, -.30, r"Kết quả báo cáo = trung bình $\pm$ độ lệch chuẩn của 5 score "
                      "→ vừa có ước lượng, vừa có mức DAO ĐỘNG",
            ha="center", va="top", fontsize=9.8, color=C3, fontweight="bold")

    # (c) stratified
    ax = axes[2]
    rng = np.random.default_rng(0)
    lab = np.r_[np.zeros(45), np.ones(5)]
    ax.text(-.015, 1.35, "Không stratified:", ha="right", va="center", fontsize=9.5)
    ax.text(-.015, .45, "Stratified:", ha="right", va="center", fontsize=9.5)
    plain = rng.permutation(lab)
    strat = np.concatenate([rng.permutation(np.r_[np.zeros(9), np.ones(1)]) for _ in range(5)])
    for row, arr, yb in [(0, plain, 1.05), (1, strat, .15)]:
        for j, v in enumerate(arr):
            ax.add_patch(Rectangle((j * .02, yb), .0145, .6,
                                   color=C2 if v == 1 else "#cbd5e1"))
        for f in range(1, 5):   # vạch chia fold nằm gọn trong khe giữa hai ô
            ax.plot([f * .2 - .0028] * 2, [yb - .05, yb + .65], color="k", lw=1.4)
    cnt = [int(plain[f * 10:(f + 1) * 10].sum()) for f in range(5)]
    fmax, fmin = int(np.argmax(cnt)) + 1, int(np.argmin(cnt)) + 1
    ax.text(1.02, 1.35, f"fold {fmax} có {max(cnt)} mẫu hiếm, fold {fmin} có {min(cnt)} "
                        "→ score giữa các fold nhảy loạn",
            va="center", fontsize=9, color=C2)
    ax.text(1.02, .45, "mỗi fold giữ ĐÚNG 1 mẫu hiếm (10%) → ước lượng ổn định",
            va="center", fontsize=9, color=C3)
    ax.set_xlim(-.16, 2.0); ax.set_ylim(0, 2.0); ax.axis("off")
    ax.set_title("Cách 3: stratified k-fold (ô đỏ = lớp hiếm)",
                 fontsize=11, fontweight="bold", loc="left")

    fig.suptitle("Ba cách chia dữ liệu để đánh giá",
                 fontweight="bold", fontsize=11.5)
    fig.tight_layout()
    save(fig, "09_kfold_cross_validation.png")


if __name__ == "__main__":
    fig_confusion_anatomy()
    fig_two_distributions()
    fig_threshold_to_roc()
    fig_pr_tradeoff()
    fig_roc_vs_pr()
    fig_accuracy_lies()
    fig_macro_micro_weighted()
    fig_calibration()
    fig_cross_validation()
    print("Xong.")
