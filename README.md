# Machine Learning: kho tài liệu và bài lab

Chào mừng đến với kho tài liệu Machine Learning được biên soạn cho sinh viên Khoa Công nghệ Thông tin, Đại học Văn Lang, chuyên ngành Trí tuệ Nhân tạo.

Mỗi bài lab gồm ba phần: **lý thuyết** giải thích "vì sao" trước "công thức gì", **thực hành** chạy được ngay, và **bài tập về nhà** có hint. Không có lời giải sẵn, sinh viên tự làm rồi báo cáo.

## Cấu trúc kho, sắp xếp từ dễ đến khó

```
Machine Learning/
├── 01. Linear_Regression/            # Hồi quy tuyến tính, Normal Eq + GD
│   ├── Linear_Regression.ipynb
│   └── images/                       # hình minh hoạ và ảnh động GIF
├── 02. Logistic_Regression/          # Hồi quy logistic, bắc cầu sang phân loại
├── 03. Naive_Bayes/                  # Phân loại xác suất, Bernoulli/Multinomial/Gaussian
├── 04. KNN/                          # K-Nearest Neighbors, scale & metric
├── 05. DecisionTree_RandomForest/    # Cây quyết định + ensemble
├── 06. SVM/                          # Support Vector Machine + kernel trick
├── 07. Accuracy_Metrics/             # Confusion matrix, Precision/Recall/F1, ROC-AUC
├── 08. K-Means/                      # Clustering không giám sát
├── 09. MLP/                          # Multi-Layer Perceptron với PyTorch
├── 10. Thi_Thu/                      # Đề thi thử
├── 11. Final_Examination/            # Đề thi cuối kỳ
└── README.md
```

Mỗi lab từ 01 đến 09 đều có thư mục `images/` theo đúng cấu trúc trên.

## Hình minh hoạ

Mỗi lab có thư mục `images/` chứa **9 đến 10 hình minh hoạ** được nhúng thẳng vào phần lý thuyết
của notebook. Hình hiện ngay trên GitHub, nên sinh viên đọc được lý thuyết mà **không cần chạy
notebook hay cài đặt gì**.

Toàn bộ hình đều được vẽ bằng matplotlib từ dữ liệu của chính bài lab đó. Không có ảnh tải từ
web nên không vướng bản quyền, không sợ link chết, và mọi con số trên hình đều khớp với phần lý
thuyết trong notebook. Ngoài hình tĩnh, mỗi lab còn có một ảnh động GIF minh hoạ quá trình
thuật toán chạy qua từng bước.

Vài hình đáng chú ý:

| Lab | Hình | Trả lời câu hỏi |
|---|---|---|
| 01 | Mặt mất mát 3D + đường đi Gradient Descent | GD thực sự "đi" thế nào trên loss surface? |
| 01 | Phép chiếu vuông góc của Normal Equation | Vì sao lại là $(X^TX)^{-1}X^Ty$? |
| 02 | BCE lồi vs MSE không lồi | Vì sao không dùng MSE cho phân loại? |
| 03 | Định lý Bayes qua 10.000 người xét nghiệm | Vì sao xét nghiệm chính xác 99% mà chỉ 17% người dương tính có bệnh? |
| 04 | Lời nguyền chiều cao | Vì sao KNN sụp đổ khi nhiều feature? |
| 05 | MDI vs permutation importance | Vì sao `feature_importances_` có thể đánh lừa bạn? |
| 06 | Kernel trick nâng chiều 3D | Vì sao dữ liệu vòng tròn lại tách được bằng mặt phẳng? |
| 07 | Từ hai phân phối điểm số đến đường ROC | ROC sinh ra từ đâu? |
| 08 | Thuật toán Lloyd từng bước | K-Means hội tụ ra sao qua từng vòng lặp? |
| 09 | XOR trong không gian ẩn | Tầng ẩn thực chất làm gì? |

## Vì sao thứ tự này?

| Bước | Lý do |
|---|---|
| 1. Linear Regression | Bài đơn giản nhất: output liên tục, mô hình tuyến tính, MSE. Là nền tảng cho mọi thứ phía sau |
| 2. Logistic Regression | Mở rộng tự nhiên của bài 1: thêm sigmoid và BCE là có bài phân loại đầu tiên |
| 3. Naive Bayes | Cách phân loại theo xác suất, tư duy khác Logistic nhưng vẫn đơn giản |
| 4. KNN | Không có training thực sự, chỉ tính khoảng cách nên rất trực quan |
| 5. Decision Tree + RF | Mô hình cây + ensemble đầu tiên |
| 6. SVM | Khái niệm margin và kernel trick, toán học sâu hơn |
| 7. Accuracy Metrics | Sau khi đã quen các model, học cách *đánh giá* model đúng |
| 8. K-Means | Bước nhảy sang học không giám sát |
| 9. MLP | Mạng neural cơ bản, cầu nối sang Deep Learning |

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
- Hiểu cơ chế của các thuật toán ML cơ bản, không dừng ở mức "biết gọi sklearn".
- Thực hiện đầy đủ một pipeline: nạp dữ liệu, tiền xử lý, chia tập, huấn luyện, đánh giá, tinh chỉnh.
- Tránh các bẫy phổ biến: data leakage, scale lẫn lộn, mất cân bằng class, sai metric.
- So sánh model với baseline (DummyClassifier, KNN, ...) trước khi kết luận.

## Yêu cầu môi trường

```bash
pip install numpy scipy pandas matplotlib seaborn scikit-learn
pip install torch torchvision      # cho lab MLP
pip install xlrd openpyxl          # đọc file .xls/.xlsx của Iris, bắt buộc cho lab KNN
pip install cvxopt                 # cho Đề thi thử (Câu 2 SVM) và bài tập SVM nâng cao
pip install imbalanced-learn       # cho bài tập class imbalance (tuỳ chọn)
```

Chỉ cần `numpy`, `scipy`, `pandas`, `matplotlib`, `scikit-learn` là chạy được các lab 01 đến 08
(không cần torch).

Khuyến nghị: Python 3.9+, dùng Jupyter Notebook hoặc Google Colab.

## Quy tắc khi nộp bài tập

1. Mỗi bài tập chỉ có **hint**, không có lời giải. Sinh viên tự code, tự kiểm.
2. Đặt `random_state=42` (hoặc seed cố định khác) để kết quả reproducible.
3. Nộp file `.ipynb` đã chạy với output, đặt tên `[HoTen]_LabX_Homework.ipynb`.
4. Mỗi bài cần kèm **markdown cell** giải thích kết quả/quan sát ngắn gọn.

## Thực hành tốt cần nhớ

| Bẫy | Cách tránh |
|---|---|
| Data leakage khi scale | `fit` scaler chỉ trên train, `transform` test |
| Accuracy lừa với class imbalance | Dùng F1, ROC-AUC, hoặc resampling |
| KNN không scale | Luôn `StandardScaler` trước KNN |
| Softmax + CrossEntropyLoss | Không đặt `nn.Softmax` ở cuối model khi dùng `nn.CrossEntropyLoss` trong PyTorch |
| Train test trên full data | Luôn `train_test_split` trước mọi bước |
| Chọn k bừa | Dùng cross-validation chứ không phải đoán mò |
