"""Chia theo bài đăng; chỉ đưa biến thể của bình luận train vào train."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
HELD_OUT_POSTS = {
    'facebook:995177076020841': 1,
    'facebook:1521398292369988': 2,
}
TRAIN_POSTS = {
    'facebook:1344727047303614', 'facebook:1514298259931518', 'facebook:3266076593542862',
    'facebook:pfbid02PTyV4AENYRGjKDktiEeiDXRxd75Rn4g4utEmEL4vfnLKt7cvjEKnXCH7feZ4XS2Gl',
    'facebook:pfbid02QHNYdMRfJ1EGmcincESkEWcbhtPCFZQFe64C4kME7YitPtWnvUrzaCbznu9FjEQxl',
    'facebook:pfbid02Vrk6YgqS35tg8ds9yEvhdbQvwqprr3vKXGYQfw7phAuZ1k2FDXL42w4MStoSJChCl',
    'facebook:pfbid02uQR9PX5LycySHi5CJP1qpwR4fxHGPmJE5bqoh92gSJeJ1dUacEUfpJboXqTzEVaHl',
    'facebook:pfbid02voZM5AHVPyeWTN5wuDPsBic65qaudo3xT3g29KkC79eXz7BbDHFQEqqhQGzmdu9al',
    'facebook:pfbid0KHrer4tBnAN3JvLdcb4yooYfXj6MXRNRWWRd4C8pKX4nWXmtL7PUXGqpQt1phnHDl',
    'facebook:pfbid0RrTKMbmrz8FQgHDQQcVnj8cVhw84MpSKdwQeFoFvVjW7WmifTyjVbAzqqEVdE4c1l',
    'facebook:pfbid0eDRFU7cHjXs3C1dwDhdok2Ap4d3SAJLmwDMCRA9CuMw6Gwdrb4yiyVnAMNdyDkJal',
    'facebook:pfbid0yjARUMvkRfNAv83RyB7YPCEs4sjv1mvMQdqkYpoFhwbqdY6at4pp2zsszQ49xYuil',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=BASE / 'data/data/pre_split/comments_preprocessed.csv')
    parser.add_argument('--output-dir', type=Path, default=BASE / 'data/data/split_all_comments/data_split_by_post')
    args = parser.parse_args()
    frame = pd.read_csv(args.input, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    flags = frame.is_synthetic.str.strip().str.lower()
    if not flags.isin(['true', 'false', '1', '0']).all():
        raise ValueError('is_synthetic phải là True/False hoặc 1/0')
    if (frame.post_id.str.strip() == '').any():
        raise ValueError('Không thể chia theo post khi thiếu post_id')
    frame['is_synthetic'] = flags.isin(['true', '1'])
    if frame.is_synthetic.any():
        raise ValueError('Bước chia chỉ nhận dữ liệu gốc; ghép synthetic sau khi chia.')
    frame['label'] = pd.to_numeric(frame.label, errors='raise').astype(int)
    real = frame.loc[~frame.is_synthetic]
    counts = pd.crosstab(real.post_id, real.label)
    if len(counts) < 3:
        raise ValueError('Cần ít nhất ba post có dữ liệu gốc')

    # Tìm cách chia gần tỷ lệ 70/20/10 cho từng nhãn của dữ liệu gốc.
    # Tỷ lệ chính xác không được đảm bảo vì không chia nhỏ post.
    values = counts.to_numpy(dtype=float)
    totals = values.sum(axis=0)
    ratios = np.array([0.7, 0.2, 0.1])
    if (set(HELD_OUT_POSTS) | TRAIN_POSTS) - set(counts.index):
        raise ValueError('Không tìm thấy đầy đủ post được cố định')
    forced_train = counts.index.isin(TRAIN_POSTS)
    if (~forced_train).sum() < 2:
        raise ValueError('Cần ít nhất hai post không có synthetic để tạo test và validation')
    rng = np.random.default_rng(42)
    best_score, best = float('inf'), None
    for _ in range(20000):
        assignment = rng.choice(3, size=len(counts), p=ratios)
        assignment[forced_train] = 0
        for post_id, split_index in HELD_OUT_POSTS.items():
            assignment[counts.index == post_id] = split_index
        if len(np.unique(assignment)) < 3:
            continue
        observed = np.array([values[assignment == i].sum(axis=0) for i in range(3)])
        score = np.square(observed / totals - ratios[:, None]).sum()
        if score < best_score:
            best_score, best = score, assignment.copy()

    post_split = dict(zip(counts.index, [ ('train', 'test', 'validation')[i] for i in best ]))
    frame['split'] = frame.post_id.map(post_split).fillna('excluded')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ('train', 'test', 'validation'):
        part = frame.loc[frame.split == name]
        filename = 'train_original.csv' if name == 'train' else f'{name}.csv'
        part.to_csv(args.output_dir / filename, index=False, encoding='utf-8-sig')
        if name == 'train':
            part.to_csv(args.output_dir / 'train.csv', index=False, encoding='utf-8-sig')
        print(f'{name}: {len(part)} dòng, {part.post_id.nunique()} post; nhãn {part.label.value_counts().to_dict()}, synthetic {part.is_synthetic.sum()}')
    print(f'Chia xong dữ liệu gốc. Thư mục: {args.output_dir}')


if __name__ == '__main__':
    main()
