# Các bước xử lý dữ liệu

Các thư mục được sắp theo công việc:

- `lam_sach/`: làm sạch bình luận crawl để gửi đi gán nhãn (`clean_data.py`).
- `tien_xu_ly/`: xử lý file đã gán nhãn, gồm kiểm tra, chuẩn hóa, tách từ và loại dòng trùng.
- `chia_tap/`: chia dữ liệu đã xử lý thành train, test, validation.

Chạy các lệnh từ thư mục gốc `BTL1-NLP`:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m xu_ly_data.preprocessing.lam_sach.clean_data
.\.venv\Scripts\python.exe -X utf8 -m xu_ly_data.preprocessing.tien_xu_ly.tach_du_lieu_dau_vao
.\.venv\Scripts\python.exe -X utf8 -m xu_ly_data.preprocessing.tien_xu_ly.tien_xu_ly_du_lieu
.\.venv\Scripts\python.exe -X utf8 -m xu_ly_data.preprocessing.chia_tap.chia_tap_du_lieu
.\.venv\Scripts\python.exe -X utf8 xu_ly_data/notebooks/run_statistics.py
```

Bước đầu tạo `xu_ly_data/data/preprocessing/01_clean.csv` để gán nhãn, không ghi đè file đã tồn tại. Dữ liệu gốc nằm trong `data/data/input/data_original.csv`; synthetic đã lọc theo nguồn train nằm trong `data/data/input/data_synthetic.csv`. Lệnh tách chỉ cập nhật dữ liệu gốc, không ghi đè file synthetic đã lọc. Tiền xử lý và chia chỉ dùng dữ liệu gốc, tạo `train_original.csv`, `train.csv`, `test.csv`, `validation.csv`. Tỷ lệ mục tiêu 70/20/10 tính trên dữ liệu gốc và từng nhãn; tỷ lệ thực tế có thể khác do kích thước post không đều.

Sau khi chia, dùng **chỉ** `train_original.csv` để sinh dữ liệu và lưu thành một CSV riêng. Hiện dùng lại `data_synthetic.csv` theo lựa chọn của người dùng, nhưng các biến thể này đã có từ trước khi chia; bước ghép chỉ giữ các biến thể có nguồn thuộc train. Mỗi dòng synthetic phải có các cột `text_raw`, `label`, `post_url`, `comment_id`, `source_comment_id`, `is_synthetic`; `post_url` và `source_comment_id` xác định bình luận gốc trong train, `comment_id` có thể giữ ID của bình luận gốc, nhãn phải trùng nguồn, và `is_synthetic` là `True`. Kiểm duyệt nội dung và nhãn trước khi sử dụng. Sau đó chạy:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m xu_ly_data.preprocessing.can_bang_du_lieu.ghep_synthetic_train --input xu_ly_data/data/data/input/data_synthetic.csv
.\.venv\Scripts\python.exe -X utf8 -m xu_ly_data.preprocessing.can_bang_du_lieu.oversampling_train
.\.venv\Scripts\python.exe -X utf8 xu_ly_data/notebooks/run_statistics.py
```

Bước ghép kiểm tra nguồn, nhãn, post và văn bản trùng. Chỉ sau khi ghép mới chạy oversampling trên train; test và validation không được tăng dữ liệu hay oversampling.

Cột `text_raw` giữ nội dung đầu vào; `text_normalized` là văn bản đã chuẩn hóa; `text_processed` là văn bản đã tách từ. Post reel `995177076020841` được cố định vào test, reel `1521398292369988` vào validation. Mười hai post còn lại được cố định vào train theo lựa chọn đã thống nhất; bước chia không đọc file synthetic. Validation và test chỉ gồm dữ liệu gốc.
