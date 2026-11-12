
import argparse
import json
from pathlib import Path

import pandas as pd

from .prepare_labeled_data import prepare_data

BASE = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=BASE / 'data/data/input/data_original.csv')
    parser.add_argument('--output-dir', type=Path, default=BASE / 'data/data/pre_split')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    real, synthetic, audit, _ = prepare_data(args.input, args.output_dir)
    if not synthetic.empty:
        raise ValueError('File đầu vào tiền xử lý chỉ được chứa dữ liệu gốc.')
    target = args.output_dir / 'comments_preprocessed.csv'
    real.to_csv(target, index=False, encoding='utf-8-sig')
    (args.output_dir / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Đã lưu {len(real)} dòng gốc trước khi chia tập: {target}')


if __name__ == '__main__':
    main()
