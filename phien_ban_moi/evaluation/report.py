"""Sinh nhận xét theo số đo và nêu rõ hạn chế nhãn/mẫu thiểu số."""
from ..training.experiments import SCENARIOS


def write_report(output, audit, split_stats, scores, best_name, baseline):
    selected = scores.set_index("model").loc[best_name]
    rows = []
    for _, row in scores.iterrows():
        rows.append(f"| {row.scenario} | {row.feature} + {row.algorithm} | {row.validation_macro_f1:.3f} | "
                    f"{row.test_accuracy:.3f} | {row.test_macro_f1:.3f} | {row.test_toxic_precision:.3f} | "
                    f"{row.test_toxic_recall:.3f} | {row.test_toxic_f1:.3f} |")
    changes = []
    for feature in ["BoW", "TF-IDF"]:
        for algorithm in ["NB", "SVM"]:
            group = scores.loc[scores.feature.eq(feature) & scores.algorithm.eq(algorithm)].set_index("scenario")
            change = group.loc["original_ros", "test_macro_f1"] - group.loc["original_imbalanced", "test_macro_f1"]
            augmented_change = group.loc["augmented_ros", "test_macro_f1"] - group.loc["original_ros", "test_macro_f1"]
            changes.append(f"- {feature} + {algorithm}: ROS làm Macro-F1 test thay đổi {change:+.3f}; "
                           f"thêm dữ liệu tổng hợp hợp lệ trên nền ROS thay đổi tiếp {augmented_change:+.3f}.")
    report = f"""# Thực nghiệm Toxic/Non Toxic về Mưa Đỏ

## Phạm vi và chất lượng nhãn

File có {audit['input_rows']} dòng: {audit['original_input']} gốc và {audit['synthetic_input']} tổng hợp.
Nhãn chữ được đổi thành 0 = Non Toxic, 1 = Toxic. Không đổi ý nghĩa hoặc tự tạo nhãn.
{audit['needs_review_original_input']} dòng gốc được file đánh dấu needs_review.
Các nhãn có nguồn đánh giá ChatGPT; kết quả này đo mức khớp nhãn được cung cấp, **chưa phải đánh giá độc lập trên ground truth con người đã xác nhận**.
Các dòng tổng hợp chưa được duyệt thủ công; xem đây là thử nghiệm tăng cường, không phải dữ liệu Facebook mới.

Sau chuẩn hóa/tách từ, dùng {audit['original_used']} bình luận gốc.
Các dòng rỗng, trùng hoặc nhãn mâu thuẫn được lưu trong `data/excluded_rows.csv`; không tự sửa nhãn mâu thuẫn.

## Flow

1. Giữ nguyên nguồn và metadata nhãn; chuẩn Unicode, viết tắt, URL, emoji thành token; tách từ bằng underthesea.
2. Cách ly mâu thuẫn, loại trùng khóa/nội dung gốc theo token vectorizer thực sự dùng.
3. Chia gốc theo bài với mục tiêu 80/10/10; chọn cách chia theo nhóm/nhãn, không theo điểm mô hình.
4. Validation/test chỉ có dữ liệu gốc, không oversampling và không có biến thể tổng hợp.
5. Tổng hợp chỉ được dùng nếu cha còn trong train, nhãn khớp cha và nội dung không trùng dữ liệu gốc.
6. BoW/TF-IDF chỉ fit trên pool train trước ROS. ROS lặp mẫu chỉ trong ma trận train.
7. Thử NB alpha = 0.1/0.5/1.0, SVM C = 0.1/1/10; unigram và unigram+bigram. Chọn bằng Macro-F1 validation.
8. Chốt lựa chọn trước khi đọc test; test dùng để báo cáo cuối, không dùng chọn mô hình/tham số.

## Cách chia thực tế

"""
    for name in ["train", "validation", "test"]:
        part = split_stats[name]
        report += f"- {name}: {part['rows']} dòng, {part['posts']} bài, số nhãn {part['labels']}.\n"
    report += f"\nCó {split_stats['synthetic_train_rows']} biến thể tổng hợp hợp lệ cho train. Các biến thể bị loại và lý do ở `data/synthetic_eligibility.csv`.\n\n"
    report += "## Xử lý mất cân bằng\n\n" + "\n".join(f"- `{key}`: {value}." for key, value in SCENARIOS.items())
    report += f"\n\nMô hình chỉ dự đoán Non Toxic đạt Accuracy {baseline['accuracy']:.4f}, Macro-F1 {baseline['macro_f1']:.4f}, Recall Toxic = 0. Vì vậy Accuracy cao không đủ để kết luận mô hình tốt.\n\n"
    report += "## Kết quả trên cùng test gốc\n\n| Cách train | Cấu hình | Val Macro-F1 | Test Accuracy | Test Macro-F1 | Toxic Precision | Toxic Recall | Toxic F1 |\n|---|---|---:|---:|---:|---:|---:|---:|\n" + "\n".join(rows)
    report += "\n\n## Nhận xét thực nghiệm\n\n" + "\n".join(changes)
    report += f"\n\nMô hình chọn bằng validation: **{best_name}**, Val Macro-F1 **{selected.validation_macro_f1:.3f}**. "
    report += f"Trên test: Macro-F1 **{selected.test_macro_f1:.3f}**, Toxic Recall **{selected.test_toxic_recall:.3f}**, Toxic F1 **{selected.test_toxic_f1:.3f}**.\n"
    report += f"\nTest chỉ có **{int(selected.test_toxic_support)} mẫu Toxic gốc**; một dự đoán thay đổi có thể làm Recall/F1 dao động mạnh. "
    report += f"Khoảng Wilson 95% tham khảo cho Recall Toxic: [{selected.test_toxic_recall_wilson_low:.3f}, {selected.test_toxic_recall_wilson_high:.3f}]. "
    report += "Khoảng này giả sử quan sát độc lập, có thể lạc quan vì bình luận cùng bài có tương quan; không phản ánh sai số nhãn.\n"
    report += "\nKhông suy ra ROS hay dữ liệu tổng hợp luôn cải thiện mô hình. Cần duyệt nhãn, bổ sung Toxic thật từ nhiều bài và giữ test độc lập nếu tiếp tục chỉnh mô hình sau khi đọc kết quả.\n"
    report += "\n## Sản phẩm\n\n`metrics.csv`: chỉ số; `validation_search.csv`: tìm tham số; `data/split_manifest.csv`: cách chia; "
    report += "`models/`: mô hình; `predictions/`: dự đoán và lỗi; `figures/`: biểu đồ; `metadata.json`: nguồn, hash, quy tắc, phiên bản. PhoBERT chưa được triển khai theo phạm vi hiện tại.\n"
    (output / "reports" / "nhan_xet_ket_qua.md").write_text(report, encoding="utf-8")
