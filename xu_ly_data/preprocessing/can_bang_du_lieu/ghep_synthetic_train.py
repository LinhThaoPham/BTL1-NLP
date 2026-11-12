"""Chỉ xử lý và ghép synthetic sau khi đã chia dữ liệu gốc theo post."""
import argparse
import hashlib
from pathlib import Path

import pandas as pd

from ..tien_xu_ly.normalize_text import load_rules, normalize_text
from ..tien_xu_ly.tokenize_text import tokenize_text
from ..tien_xu_ly.prepare_labeled_data import LABEL_MAP, feature_key, post_identity

BASE = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='File synthetic mới được sinh từ train_original.csv sau khi chia')
    parser.add_argument('--split-dir', type=Path, default=BASE / 'data/data/split_all_comments/data_split_by_post')
    args = parser.parse_args()
    parts = {name: pd.read_csv(args.split_dir / (('train_original' if name == 'train' else name) + '.csv'), dtype=str, keep_default_na=False, encoding='utf-8-sig') for name in ('train', 'test', 'validation')}
    posts = {name: set(part.post_id) for name, part in parts.items()}
    if any(posts[a] & posts[b] for a, b in [('train', 'test'), ('train', 'validation'), ('test', 'validation')]):
        raise ValueError('Post xuất hiện trong nhiều tập.')
    synth = pd.read_csv(args.input, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    if not synth.is_synthetic.str.strip().str.lower().isin(['true', '1']).all():
        raise ValueError('File synthetic chỉ được chứa dữ liệu synthetic.')
    synth['post_id'] = synth.post_url.map(post_identity)
    synth['parent_source_key'] = synth.post_id + '|' + synth.source_comment_id
    synth['source_key'] = synth.post_id + '|' + synth.comment_id
    synth['input_row'] = range(2, len(synth) + 2)
    synth['record_id'] = 'synthetic:' + synth.input_row.astype(str)
    synth['label_original'] = synth.label
    synth['label'] = synth.label.str.strip().str.lower().map(LABEL_MAP)
    if synth.label.isna().any():
        raise ValueError('Nhãn synthetic không hợp lệ.')
    synth['excluded_reason'] = ''
    synth.loc[~synth.post_id.isin(posts['train']), 'excluded_reason'] = 'post_not_in_train'
    train_keys = set(parts['train'].source_key)
    synth.loc[(synth.excluded_reason == '') & ~synth.parent_source_key.isin(train_keys), 'excluded_reason'] = 'source_not_in_train'
    source_labels = parts['train'].set_index('source_key').label
    source_label = synth.parent_source_key.map(source_labels)
    synth.loc[(synth.excluded_reason == '') & (synth.label.astype(str) != source_label), 'excluded_reason'] = 'source_label_mismatch'
    candidates = synth.loc[synth.excluded_reason == ''].copy()
    rules = load_rules()
    candidates['text_clean'] = candidates.text_raw.map(lambda text: ' '.join(text.split()))
    candidates['text_normalized'] = candidates.text_clean.map(lambda text: normalize_text(text, rules))
    candidates['text_processed'] = candidates.text_normalized.map(lambda text: tokenize_text(text, rules))
    candidates['text_key'] = candidates.text_processed.map(feature_key)
    candidates['text_hash'] = candidates.text_key.map(lambda text: hashlib.sha256(text.encode()).hexdigest())
    existing = set(pd.concat([part.text_key for part in parts.values()]))
    dup = candidates.text_key.eq('') | candidates.text_key.isin(existing) | candidates.duplicated('text_key')
    synth.loc[candidates.index[dup], 'excluded_reason'] = 'duplicate_or_empty_content'
    accepted = candidates.loc[~dup].drop(columns='excluded_reason')
    accepted['split'] = 'train'
    result = pd.concat([parts['train'], accepted], ignore_index=True)
    result.to_csv(args.split_dir / 'train.csv', index=False, encoding='utf-8-sig')
    synth.loc[synth.excluded_reason != ''].to_csv(args.split_dir / 'synthetic_excluded.csv', index=False, encoding='utf-8-sig')
    print(f"Train: {len(parts['train'])} gốc + {len(accepted)} synthetic = {len(result)}; loại {sum(synth.excluded_reason != '')} synthetic")


if __name__ == '__main__':
    main()
