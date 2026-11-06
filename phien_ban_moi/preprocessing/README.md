# Tiền xử lý bình luận Mưa Đỏ

Phần này độc lập với code crawl. Chỉ đọc CSV đầu vào, không sửa file ở `crawl data`.

## Từng file làm gì?

| File | Nhiệm vụ | Đầu ra mặc định |
|---|---|---|
| `clean_data.py` | Chuẩn Unicode/khoảng trắng, bỏ dòng rỗng và trùng ID | `data/01_clean.csv` |
| `normalize_text.py` | Chữ thường, chuẩn từ viết tắt, URL/email/mention, emoji thành token | `data/02_normalized.csv` |
| `tokenize_text.py` | Tách từ bằng underthesea, tùy chọn loại stop words | `data/03_processed.csv` |
| `split_data.py` | Kiểm tra nhãn, loại trùng nội dung, chia theo bài | `data/splits/train.csv`, `validation.csv`, `test.csv` |
| `common.py` | Hàm đọc/ghi CSV và kiểm tra cột dùng chung | Không có |
| `text_rules.json` | Quy tắc có thể chỉnh, không có mô hình hoặc luật tự gán nhãn | Không có |

## Chạy theo thứ tự

Mở terminal tại thư mục dự án. Dùng môi trường đã cài dependencies ở `requirements.txt`:

```powershell
cd "phien_ban_moi\preprocessing"
python clean_data.py
```

**Dừng tại đây để gán nhãn:** mở `data/01_clean.csv`, đọc `text_raw` và điền cột `label`:
`0 = Non Toxic`, `1 = Toxic`. Giữ nguyên ID và link nguồn. Không tự tạo nhãn cho đủ số lượng.
Nếu đã có file gán nhãn từ script cũ, có thể chạy bước 1 với
`--input "..\crawl data\comments_with_label.csv"` và một tên `--output` mới.

Sau khi đã lưu nhãn:

```powershell
python normalize_text.py
python tokenize_text.py
python split_data.py
```

Mặc định giữ stop words; muốn thử loại chúng, dùng `python tokenize_text.py --remove-stopwords`
với `--output` khác. Phủ định và từ tục trong danh sách bảo vệ vẫn giữ.
Không tự loại bình luận ngắn hoặc chê phim. Người duyệt tự kiểm tra các dòng lấy nhầm giao diện
hoặc không liên quan trước khi gán nhãn.

## Ý nghĩa các cột

- `text_raw`: nguyên văn, để gán/duyệt nhãn.
- `text_clean`: làm sạch cơ bản, giữ chữ hoa, dấu câu, emoji, từ tục.
- `text_normalized`: chuẩn hóa cho mô hình truyền thống, ví dụ `ko` thành `không`, emoji thành `emoji_...`.
- `text_processed`: đã tách từ; dùng làm đầu vào BoW/TF-IDF.
- `label`: nhãn người gán. Các bước xử lý không đổi hoặc tự điền nhãn.
- `post_url`, `comment_id`, `parent_comment_id`: metadata nếu có ở đầu vào.
- `post_id`: thêm ở bước chia tập để giữ cùng bài trong cùng một tập.

## Chia tập và bàn giao

Tỷ lệ mục tiêu train/validation/test là 80/10/10, có thể lệch vì phải giữ nguyên nhóm bài viết.
Cần nhiều bài có cả hai nhãn; chương trình báo lỗi nếu không chia được, không chuyển sang chia ngẫu nhiên từng dòng.
Nội dung giống nhau sau xử lý được giữ một bản; nhãn mâu thuẫn phải duyệt lại.
Các mô hình dùng chung ba file chia. Chỉ fit từ vựng/IDF trên train; chọn tham số trên validation,
đánh giá cuối trên test. Không cân bằng/oversample validation và test.

`text_processed` dành cho BoW/TF-IDF. Người làm PhoBERT dùng `text_clean` của đúng các hàng đã chia,
rồi tách từ và tokenize theo đầu vào PhoBERT; không tự dùng tokenizer TF-IDF cho PhoBERT.

Các script không ghi đè đầu ra đã tồn tại nhằm tránh mất nhãn hoặc thay đổi tập test đã chốt.
Muốn chạy lại, chọn `--output`/`--output-dir` mới; bước sau truyền `--input` tương ứng.
