"""Tách dữ liệu gốc từ file hỗn hợp, giữ nguyên file synthetic đã lọc."""
import argparse
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=BASE / 'data/data/input/data_labeled.csv')
    parser.add_argument('--output-dir', type=Path, default=BASE / 'data/data/input')
    args = parser.parse_args()
    data = pd.read_csv(args.input, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    flags = data.is_synthetic.str.strip().str.lower()
    if not flags.isin(['true', 'false', '1', '0']).all():
        raise ValueError('is_synthetic phải là True/False hoặc 1/0')
    synthetic = flags.isin(['true', '1'])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    original = data.loc[~synthetic]
    original.to_csv(args.output_dir / 'data_original.csv', index=False, encoding='utf-8-sig')
    print(f'data_original.csv: {len(original)} dòng; giữ nguyên file synthetic đã lọc')


if __name__ == '__main__':
    main()
