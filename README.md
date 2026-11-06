# Phân loại Toxic/Non Toxic về phim Mưa Đỏ

Code và dữ liệu đang sử dụng nằm trong `phien_ban_moi/`.

## Dữ liệu sau tiền xử lý để mở và trình bày

- [comments_preprocessed.csv](phien_ban_moi/data/processed/comments_preprocessed.csv): toàn bộ dữ liệu sau tiền xử lý, trước BoW/TF-IDF và trước oversampling; cột `text_processed` là đầu vào mô hình, cột `split` xác định tập đã chia.
- `phien_ban_moi/data/processed/train.csv`, `validation.csv`, `test.csv`: ba tập tương ứng.

Lệnh `python -m phien_ban_moi.training.run_full_input --input DUONG_DAN_FILE_NHAN.csv` cập nhật các file này. Thư mục có thời gian trong `results/` lưu kết quả riêng từng lần chạy để đối chiếu.

## Cấu trúc

- `phien_ban_moi/crawl data/`: lấy link bài viết, crawl bình luận, cookie cục bộ.
- `phien_ban_moi/preprocessing/`: làm sạch, chuẩn hóa, tách từ và chia tập.
- `phien_ban_moi/training/`: BoW/TF-IDF × Naive Bayes/SVM; oversampling và tăng cường train.
- `phien_ban_moi/evaluation/`: chỉ số, biểu đồ và nhận xét.
- `phien_ban_moi/notebooks/02_thuc_nghiem_nb_svm.ipynb`: đọc kết quả thực nghiệm đã chạy.
- `phien_ban_moi/01_khao_sat_va_nhan_xet_du_lieu.ipynb`: khảo sát dữ liệu crawl.
- `phien_ban_moi/data/input/`: snapshot đầu vào theo SHA256.
- `phien_ban_moi/results/`: kết quả từng lần; `latest.json` trỏ tới lần hoàn thành gần nhất.
- `phien_ban_moi/tests/`: kiểm thử cho code hiện tại.
- `phien_ban_moi/_archive/`: bản ZIP khôi phục code/tài liệu/dữ liệu phiên bản cũ.

## Chạy

Mở terminal tại thư mục `BTL1-NLP`. Chọn Python 3.12 trong `.venv`.

```powershell
& '.\.venv\Scripts\python.exe' -m phien_ban_moi.training.run_all --input 'DUONG_DAN_FILE_NHAN.csv'
```

Đọc hướng dẫn đầy đủ trong [README thực nghiệm](phien_ban_moi/README_THUC_NGHIEM.md).
Dự đoán bằng model do dự án tạo ra:

```powershell
& '.\.venv\Scripts\python.exe' -m phien_ban_moi.training.predict --model 'DUONG_DAN_MODEL.joblib' --text 'Phim này chưa hay.'
```

Kiểm thử:

```powershell
& '.\.venv\Scripts\python.exe' -m pytest -q
```

Giữ `.venv`, `.git`, `requirements.txt`, `requirements-lock.txt` và `pyproject.toml` tại thư mục gốc.
Không gửi cookie hoặc mô hình joblib không rõ nguồn. Các cờ nguồn dữ liệu và nhãn cần duyệt vẫn được giữ.
Ở lần chạy `run_full_input`, toàn bộ đầu vào được chia theo bài viết; oversampling chỉ áp dụng trên train.
PhoBERT chưa nằm trong phạm vi thực nghiệm hiện tại.
