# Dữ liệu train sau oversampling

`train_oversampled.csv` được tạo từ `../data_split_by_post/train.csv` bằng cách lấy lặp ngẫu nhiên các dòng toxic đến khi số lượng bằng non-toxic; seed là 42. Cột `is_oversampled` đánh dấu các dòng lấy lặp thêm. Những dòng này không phải bình luận mới.

Chạy từ thư mục gốc BTL1-NLP:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m xu_ly_data.preprocessing.can_bang_du_lieu.oversampling_train
```

Lệnh ghi lại file kết quả mặc định; dùng `--output` để chọn file khác. Validation và test vẫn dùng các file trong `../data_split_by_post/`.
