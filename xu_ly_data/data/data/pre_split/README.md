# Tiền xử lý trước khi chia tập

Chạy từ thư mục BTL1-NLP:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m xu_ly_data.preprocessing.tien_xu_ly.tien_xu_ly_du_lieu
```

Lệnh tạo `comments_preprocessed.csv` trước khi chia. File gồm bình luận thu thập và các biến thể được đánh dấu `is_synthetic`. Các dòng trùng nội dung sau chuẩn hóa được loại kể cả khác ID; nội dung có nhãn mâu thuẫn được lưu riêng để xem lại.

Chạy `.\.venv\Scripts\python.exe -X utf8 -m xu_ly_data.preprocessing.chia_tap.chia_tap_du_lieu` để chia toàn bộ bình luận và biến thể trong từng bài theo tỷ lệ mục tiêu train 70%, test 20%, validation 10% (làm tròn với bài ít bình luận). Kết quả nằm ở `data/data/split_all_comments/data_split_original/`. Các câu nguồn và biến thể có thể nằm khác tập, ảnh hưởng kết quả đánh giá.
