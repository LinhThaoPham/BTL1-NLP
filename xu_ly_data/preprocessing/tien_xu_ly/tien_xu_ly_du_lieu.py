
import argparse
import json
from pathlib import Path

import pandas as pd

from .prepare_labeled_data import prepare_data

BASE = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=BASE / 'data/data/input/data_labeled.csv')
    parser.add_argument('--output-dir', type=Path, default=BASE / 'data/data/pre_split')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    real, synthetic, audit, _ = prepare_data(args.input, args.output_dir)
    combined = pd.concat([real, synthetic], ignore_index=True).sort_values('input_row')
    conflicting = combined.groupby('text_key').label.transform('nunique').gt(1)
    duplicates = ~conflicting & combined.duplicated('text_key')
    rejected = combined.loc[conflicting | duplicates].copy()
    rejected['excluded_reason'] = 'duplicate_content'
    rejected.loc[conflicting.loc[rejected.index], 'excluded_reason'] = 'conflicting_content_labels'
    rejected.to_csv(args.output_dir / 'excluded_combined.csv', index=False, encoding='utf-8-sig')
    audit['combined_exclusions'] = rejected.excluded_reason.value_counts().to_dict()
    combined = combined.loc[~(conflicting | duplicates)]
    target = args.output_dir / 'comments_preprocessed.csv'
    combined.to_csv(target, index=False, encoding='utf-8-sig')
    (args.output_dir / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Đã lưu {len(combined)} dòng trước khi chia tập: {target}')


if __name__ == '__main__':
    main()
