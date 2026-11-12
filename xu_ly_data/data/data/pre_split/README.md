# Tiền xử lý trước khi chia tập

Chạy từ thư mục BTL1-NLP:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m xu_ly_data.preprocessing.tien_xu_ly.tien_xu_ly_du_lieu
```

Lệnh đọc `data_original.csv` và tạo `comments_preprocessed.csv` chỉ chứa bình luận gốc trước khi chia. Các dòng trùng nội dung sau chuẩn hóa được loại kể cả khác ID; nội dung có nhãn mâu thuẫn được lưu riêng để xem lại.

Chạy `.\.venv\Scripts\python.exe -X utf8 -m xu_ly_data.preprocessing.chia_tap.chia_tap_du_lieu` để chia các post gốc. Chỉ sau khi chia mới sinh dữ liệu từ `train_original.csv`, rồi ghép file mới bằng `ghep_synthetic_train --input <file_moi.csv>` và oversampling train. Kết quả nằm ở `data/data/split_all_comments/data_split_by_post/`.
