# Dữ liệu sau tiền xử lý

Mở `comments_preprocessed.csv` để xem toàn bộ dữ liệu đã làm sạch, chuẩn hóa và tách từ, trước BoW/TF-IDF và trước oversampling.

- `text_raw`: nội dung đầu vào.
- `text_normalized`: nội dung đã chuẩn hóa.
- `text_processed`: nội dung đã tách từ, dùng cho BoW/TF-IDF.
- `label`: 0 = non-toxic; 1 = toxic.
- `split`: train, validation hoặc test.

`train.csv`, `validation.csv`, `test.csv` giữ nguyên cách chia của lần thực nghiệm. Không dùng file gộp để huấn luyện rồi đánh giá lại trên test. Oversampling chỉ thực hiện trên train trong bước huấn luyện.

Các file được cập nhật bởi `phien_ban_moi.training.run_full_input`. Bản kết quả từng lần vẫn nằm trong `results/`.
