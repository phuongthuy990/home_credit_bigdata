# Home Credit Default Risk – Big Data Project

Đồ án môn Big Data sử dụng **Apache Spark (PySpark)** để phân tích và dự đoán
khả năng vỡ nợ của khách hàng từ bộ dữ liệu
[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) trên Kaggle.

---

## Cấu trúc thư mục

```
home_credit_bigdata/
├── data/                       # Dữ liệu thô từ Kaggle (CSV)
├── notebooks/                  # Jupyter Notebooks theo từng thành viên
├── src/                        # Source code PySpark
│   ├── cleaning.py             # Thành viên 1
│   ├── feature_engineering.py  # Thành viên 2
│   ├── join_tables.py          # Thành viên 3
│   └── train_model.py          # Huấn luyện mô hình
├── output/
│   ├── cleaned_data/           # Parquet sau làm sạch
│   └── final_features.parquet  # Features tổng hợp cuối cùng
├── reports/
│   └── EDA_visualizations/     # Biểu đồ, hình ảnh phân tích
├── requirements.txt
└── README.md
```

---

## Phân công công việc

| Thành viên | Nhiệm vụ | File |
|------------|----------|------|
| Thành viên 1 | Data Cleaning – xử lý null, anomaly | `src/cleaning.py` · `notebooks/member1_data_cleaning.ipynb` |
| Thành viên 2 | Feature Engineering – GROUP BY, aggregations | `src/feature_engineering.py` · `notebooks/member2_feature_engineering.ipynb` |
| Thành viên 3 | JOIN Optimization – Broadcast Join | `src/join_tables.py` · `notebooks/member3_join_optimization.ipynb` |
| Cả nhóm | Huấn luyện mô hình | `src/train_model.py` |

---

## Cài đặt môi trường

### Yêu cầu
- Python ≥ 3.10
- Java 8 hoặc 11 (bắt buộc cho Spark)
- Apache Spark 3.5.x

### Cài thư viện

```bash
pip install -r requirements.txt
```

### Tải dữ liệu từ Kaggle

```bash
kaggle competitions download -c home-credit-default-risk
# Giải nén vào thư mục data/
```

---

## Chạy pipeline

### 1. Làm sạch dữ liệu

```bash
python src/cleaning.py
```

### 2. Tạo features tổng hợp + JOIN

```bash
python src/join_tables.py
```

### 3. Huấn luyện mô hình

```bash
python src/train_model.py
```

---

## Kỹ thuật Spark được áp dụng

| Kỹ thuật | Mô tả |
|----------|-------|
| **Broadcast Join** | Giảm shuffle khi JOIN bảng nhỏ với bảng lớn |
| **Parquet + Snappy** | Lưu dữ liệu dạng cột, nén hiệu quả |
| **Lazy Evaluation** | Spark chỉ thực thi khi có action (count, write…) |
| **approxQuantile** | Tính median xấp xỉ nhanh cho fill null |
| **MLlib Pipeline** | Chuỗi Imputer → Assembler → Scaler → Classifier |

---

## Mô hình Machine Learning

- **Logistic Regression** (baseline)
- **Random Forest Classifier** (ensemble)
- Đánh giá bằng **AUC-ROC** trên tập test (80/20 split)
