# Các bước xử lý dữ liệu

Các thư mục được sắp theo công việc:

- `lam_sach/`: làm sạch bình luận crawl để gửi đi gán nhãn (`clean_data.py`).
- `tien_xu_ly/`: xử lý file đã gán nhãn, gồm kiểm tra, chuẩn hóa, tách từ và loại dòng trùng.
- `chia_tap/`: chia dữ liệu đã xử lý thành train, test, validation.

Chạy các lệnh từ thư mục gốc `BTL1-NLP`:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m xu_ly_data.preprocessing.lam_sach.clean_data
.\.venv\Scripts\python.exe -X utf8 -m xu_ly_data.preprocessing.tien_xu_ly.tien_xu_ly_du_lieu
.\.venv\Scripts\python.exe -X utf8 -m xu_ly_data.preprocessing.chia_tap.chia_tap_du_lieu
```

Bước đầu tạo `xu_ly_data/data/preprocessing/01_clean.csv` để gán nhãn, không ghi đè file đã tồn tại. Sau khi có file gán nhãn `xu_ly_data/data/data/input/data_labeled.csv`, chạy bước tiền xử lý để tạo `data/data/pre_split/comments_preprocessed.csv`. Bước chia tạo ba file trong `data/data/split_all_comments/data_split_original/` với tỷ lệ mục tiêu 70% train, 20% test, 10% validation trong từng bài. Hai bước sau ghi lại đầu ra mặc định nếu chạy lại; dùng `--output-dir` để lưu riêng.

Cột `text_raw` giữ nội dung đầu vào; `text_normalized` là văn bản đã chuẩn hóa; `text_processed` là văn bản đã tách từ. Câu nguồn và biến thể có thể nằm khác tập, cần lưu ý khi diễn giải kết quả đánh giá.
