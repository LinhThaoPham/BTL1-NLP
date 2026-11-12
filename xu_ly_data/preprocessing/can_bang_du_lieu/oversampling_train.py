import argparse
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=BASE / 'data/data/split_all_comments/data_split_by_post/train.csv')
    parser.add_argument('--output', type=Path, default=BASE / 'data/data/split_all_comments/data_oversampling/train_oversampled.csv')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    if 'split' not in df or not df['split'].eq('train').all():
        raise ValueError('Đầu vào phải chỉ gồm các dòng thuộc tập train.')
    if not df['label'].isin(['0', '1']).all():
        raise ValueError('Nhãn phải là 0 hoặc 1.')
    if 'is_synthetic' not in df or not df['is_synthetic'].str.strip().str.lower().isin(['true', '1']).any():
        raise ValueError('Hãy sinh dữ liệu từ train và ghép synthetic mới trước khi oversampling.')
    toxic = df[df['label'] == '1']
    non_toxic = df[df['label'] == '0']
    if toxic.empty or non_toxic.empty or len(toxic) > len(non_toxic):
        raise ValueError('Cần hai lớp và số toxic không được lớn hơn non-toxic.')

    df['is_oversampled'] = 'False'
    additional = toxic.sample(n=len(non_toxic) - len(toxic), replace=True, random_state=args.seed).copy()
    additional['is_oversampled'] = 'True'
    balanced = pd.concat([df, additional], ignore_index=True).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    balanced.to_csv(args.output, index=False, encoding='utf-8-sig')

    print(f'Train trước oversampling: {len(df)} dòng; non-toxic={len(non_toxic)}, toxic={len(toxic)}')
    print(f'Train sau oversampling: {len(balanced)} dòng; non-toxic={len(non_toxic)}, toxic={len(non_toxic)}')
    print(f'Số dòng lấy lặp thêm: {len(additional)}. Đã lưu: {args.output}')


if __name__ == '__main__':
    main()
