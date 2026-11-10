
import argparse
from pathlib import Path

import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=BASE / 'data/data/pre_split/comments_preprocessed.csv')
    parser.add_argument('--output-dir', type=Path, default=BASE / 'data/data/split_all_comments/data_split_original')
    args = parser.parse_args()
    frame = pd.read_csv(args.input, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    flags = frame.is_synthetic.str.lower()
    if not flags.isin(['true', 'false', '1', '0']).all():
        raise ValueError('is_synthetic phải là True/False hoặc 1/0')
    frame['is_synthetic'] = flags.isin(['true', '1'])
    frame['label'] = pd.to_numeric(frame.label, errors='raise').astype(int)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    groups = {'train': [], 'test': [], 'validation': []}
    for _, post in frame.groupby('post_id', sort=True):
        indices = rng.permutation(post.index.to_numpy())
        expected = len(indices) * np.array([0.7, 0.2, 0.1])
        counts = np.floor(expected).astype(int)
        order = np.argsort(-(expected - counts), kind='stable')
        counts[order[:len(indices) - counts.sum()]] += 1
        train_end, test_end = counts[0], counts[0] + counts[1]
        for name, idx in zip(groups, (indices[:train_end], indices[train_end:test_end], indices[test_end:])):
            groups[name].extend(idx)
    pieces = {name: frame.loc[indices].assign(split=name) for name, indices in groups.items()}
    for name, part in pieces.items():
        part.to_csv(args.output_dir / f'{name}.csv', index=False, encoding='utf-8-sig')
    print('Đã chia:', {name: len(part) for name, part in pieces.items()})
    print(f'Thư mục: {args.output_dir}')


if __name__ == '__main__':
    main()
