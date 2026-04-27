# Machine Learning — Kho tài liệu và Bài Lab

Chào mừng đến với kho tài liệu Machine Learning được biên soạn cho sinh viên Khoa Công nghệ Thông tin, Đại học Văn Lang, chuyên ngành Trí tuệ Nhân tạo.

Mỗi bài lab gồm ba phần: **lý thuyết** giải thích "vì sao" trước "công thức gì", **thực hành** chạy được ngay, và **bài tập về nhà** có hint (không có lời giải — sinh viên tự làm và báo cáo).

## Cấu trúc kho — sắp xếp từ dễ đến khó

```
Machine Learning/
├── 1. Linear_Regression/             # Hồi quy tuyến tính, Normal Eq + GD
├── 2. Logistic_Regression/           # Hồi quy logistic — bắc cầu sang phân loại
├── 3. Naive_Bayes/                   # Phân loại xác suất, Bernoulli/Multinomial/Gaussian
├── 4. KNN/                           # K-Nearest Neighbors, scale & metric
├── 5. DecisionTree_RandomForest/     # Cây quyết định + ensemble
├── 6. SVM/                           # Support Vector Machine + kernel trick
├── 7. Accuracy_Metrics/              # Confusion matrix, Precision/Recall/F1, ROC-AUC
├── 8. K-Means/                       # Clustering không giám sát
├── 9. MLP/                           # Multi-Layer Perceptron với PyTorch
├── 10. Thi_Thu/                      # Đề thi thử
├── 11. Final_Examination/            # Đề thi cuối kỳ
└── README.md
```

## Vì sao thứ tự này?

| Bước | Lý do |
|---|---|
| 1. Linear Regression | Bài đơn giản nhất — output liên tục, mô hình tuyến tính, MSE. Là nền tảng cho mọi thứ phía sau |
| 2. Logistic Regression | Mở rộng tự nhiên: thêm sigmoid + BCE → bài phân loại đầu tiên |
| 3. Naive Bayes | Cách phân loại theo xác suất — tư duy khác Logistic, vẫn đơn giản |
| 4. KNN | Không có training thực sự, chỉ tính khoảng cách — trực quan |
| 5. Decision Tree + RF | Mô hình cây + ensemble đầu tiên |
| 6. SVM | Khái niệm margin + kernel trick — toán học sâu hơn |
| 7. Accuracy Metrics | Sau khi đã quen các model, học cách *đánh giá* model đúng |
| 8. K-Means | Bước nhảy sang học không giám sát |
| 9. MLP | Mạng neural cơ bản — cầu nối sang Deep Learning |

## Lộ trình 10 tuần đề xuất

| Tuần | Bài | Số tiết |
|---|---|---|
| 1 | Linear Regression | 3 |
| 2 | Logistic Regression | 3 |
| 3 | Naive Bayes | 3 |
| 4 | KNN | 3 |
| 5 | Decision Tree + Random Forest | 3 |
| 6 | SVM | 3 |
| 7 | Accuracy / Metrics | 3 |
| 8 | K-Means | 3 |
| 9 | MLP | 3 |
| 10 | Ôn tập + Thi cuối kỳ | 3 |

## Mục tiêu môn học

Sau khi hoàn thành, sinh viên có thể:
- Hiểu cơ chế của các thuật toán ML cơ bản — không chỉ "biết dùng sklearn".
- Thực hiện đầy đủ pipeline: load → preprocess → split → train → evaluate → tune.
- Tránh các bẫy phổ biến: data leakage, scale lẫn lộn, mất cân bằng class, sai metric.
- So sánh model với baseline (DummyClassifier, KNN, ...) trước khi kết luận.

## Yêu cầu môi trường

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
pip install torch torchvision      # cho lab MLP
pip install xlrd openpyxl          # đọc file .xls/.xlsx của Iris
pip install cvxopt                 # cho Đề thi thử (Câu 2 SVM) và bài tập SVM nâng cao
pip install imbalanced-learn       # cho bài tập class imbalance (tuỳ chọn)
```

Khuyến nghị: Python 3.9+, dùng Jupyter Notebook hoặc Google Colab.

## Quy tắc khi nộp bài tập

1. Mỗi bài tập có **hint** — không có lời giải. Sinh viên tự code, tự kiểm.
2. Đặt `random_state=42` (hoặc seed cố định khác) để kết quả reproducible.
3. Nộp file `.ipynb` đã chạy với output, đặt tên `[HoTen]_LabX_Homework.ipynb`.
4. Mỗi bài cần kèm **markdown cell** giải thích kết quả/quan sát ngắn gọn.

## Thực hành tốt cần nhớ

| Bẫy | Cách tránh |
|---|---|
| Data leakage khi scale | `fit` scaler chỉ trên train, `transform` test |
| Accuracy lừa với class imbalance | Dùng F1, ROC-AUC, hoặc resampling |
| KNN không scale | Luôn `StandardScaler` trước KNN |
| Softmax + CrossEntropyLoss | KHÔNG đặt `nn.Softmax` cuối model trong PyTorch |
| Train test trên full data | Luôn `train_test_split` trước mọi bước |
| Chọn k bừa | Dùng cross-validation chứ không phải đoán mò |
