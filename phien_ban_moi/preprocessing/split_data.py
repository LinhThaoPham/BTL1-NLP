"""Bước 4: kiểm tra nhãn và chia train/validation/test theo bài, xấp xỉ 80/10/10."""
import argparse
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from sklearn.model_selection import GroupShuffleSplit

from common import DATA_DIR, read_csv, require_columns, run_cli, write_csv


def post_group(url):
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    if query.get("story_fbid"):
        return "facebook:" + query["story_fbid"][0]
    parts = parsed.path.strip("/").split("/")
    for section in ("posts", "videos", "reel", "permalink"):
        if section in parts and parts.index(section) + 1 < len(parts):
            return "facebook:" + parts[parts.index(section) + 1]
    return url.strip()


def split_data(frame, seed=42):
    require_columns(frame, ["text_raw", "text_processed", "label", "post_url"])
    frame = frame.copy()
    frame["label"] = frame["label"].astype(str).str.strip()
    if not frame["label"].isin(["0", "1"]).all():
        raise ValueError("Còn nhãn trống/không hợp lệ. Hoàn thành gán nhãn 0 hoặc 1 trước khi chia tập.")
    if frame.empty or set(frame["label"]) != {"0", "1"}:
        raise ValueError("Cần dữ liệu có đủ hai nhãn Toxic và Non Toxic.")
    if frame["post_url"].str.strip().eq("").any() or frame["text_processed"].str.strip().eq("").any():
        raise ValueError("Còn nguồn hoặc văn bản xử lý rỗng; kiểm tra dữ liệu trước khi chia.")
    keys = frame["text_processed"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    frame["_text_key"] = keys
    if (frame.groupby("_text_key")["label"].nunique() > 1).any():
        raise ValueError("Nội dung giống nhau sau xử lý nhưng nhãn khác nhau; cần duyệt lại.")
    frame = frame.drop_duplicates("_text_key").reset_index(drop=True)
    frame["post_id"] = frame["post_url"].map(post_group)
    if frame["post_id"].nunique() < 6:
        raise ValueError("Cần ít nhất 6 bài khác nhau sau loại trùng; nên có 10 bài trở lên để gần 80/10/10.")
    best, best_score = None, float("inf")
    # Thử nhiều cách chia chỉ dựa trên nhóm/nhãn, không dựa vào điểm mô hình.
    for attempt in range(100):
        outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed + attempt)
        train_idx, holdout_idx = next(outer.split(frame, groups=frame["post_id"]))
        holdout = frame.iloc[holdout_idx]
        if holdout["post_id"].nunique() < 2:
            continue
        inner = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed + attempt)
        val_rel, test_rel = next(inner.split(holdout, groups=holdout["post_id"]))
        pieces = [frame.iloc[train_idx], holdout.iloc[val_rel], holdout.iloc[test_rel]]
        if any(set(piece["label"]) != {"0", "1"} for piece in pieces):
            continue
        score = sum(abs(len(piece) / len(frame) - target) for piece, target in zip(pieces, [0.8, 0.1, 0.1]))
        if score < best_score:
            best, best_score = pieces, score
    if best is None:
        raise ValueError("Chưa chia được 3 tập đều có hai lớp; cần thêm nhãn ở nhiều bài khác nhau.")
    return dict(zip(["train", "validation", "test"], [part.drop(columns="_text_key").reset_index(drop=True) for part in best]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATA_DIR / "03_processed.csv")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR / "splits")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    frame = read_csv(args.input)
    pieces = split_data(frame, args.seed)
    # Kiểm tra toàn bộ tên đầu ra trước khi ghi bất kỳ tập nào.
    for name in pieces:
        if (args.output_dir / f"{name}.csv").exists():
            raise ValueError("Tập chia đã tồn tại; chọn --output-dir mới để giữ cách chia cũ.")
    for name, piece in pieces.items():
        write_csv(piece, args.output_dir / f"{name}.csv")
        print(f"{name}: {len(piece)} dòng, {piece.post_id.nunique()} bài, nhãn {piece.label.value_counts().to_dict()}")
    print("NB/SVM/PhoBERT dùng chung các hàng đã chia. TF-IDF/BoW chỉ fit trên train.")


if __name__ == "__main__":
    run_cli(main)
