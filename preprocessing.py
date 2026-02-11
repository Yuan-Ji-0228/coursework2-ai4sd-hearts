#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WinoBias Preprocessing for Stereotype Detection (Group Split by document_id)

Key fix:
- Split by document_id to avoid pair leakage (pro/anti variants must stay in the same split)
"""

import argparse
import os
import hashlib
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import GroupShuffleSplit

def load_winobias_data():
    print("[INFO] Loading WinoBias dataset...")

    datasets = {}
    configs = ['type1_anti', 'type1_pro', 'type2_anti', 'type2_pro']

    for config in configs:
        try:
            ds = load_dataset("uclanlp/wino_bias", config)
            datasets[config] = ds
            total_samples = sum(len(split) for split in ds.values())
            print(f"   Loaded {config}: {total_samples} samples")
        except Exception as e:
            print(f"   Failed to load {config}: {e}")

    return datasets

def _stable_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def process_winobias_samples(datasets):
    rows = []

    for config_name, dataset in datasets.items():
        is_stereotype = 'pro' in config_name
        label = 1 if is_stereotype else 0

        dataset_type = 'type1' if 'type1' in config_name else 'type2'
        stype = 'pro' if is_stereotype else 'anti'

        print(f"\n[INFO] Processing {config_name}...")

        sample_count = 0
        for split_name in dataset.keys():
            for sample in dataset[split_name]:
                if 'tokens' in sample and sample['tokens']:
                    tokens = sample['tokens']
                    if isinstance(tokens, list):
                        sentence = ' '.join(tokens).strip()
                        if len(sentence) == 0:
                            continue

                        doc_id = sample.get('document_id', '')
                        rows.append({
                            'sentence': sentence,
                            'label': label,
                            'source': 'winobias',
                            'bias_type': 'gender',
                            'dataset_type': dataset_type,
                            'stereotype_type': stype,
                            'document_id': doc_id
                        })
                        sample_count += 1

        print(f"  Added {sample_count} samples")
        if sample_count > 0:
            examples = [r for r in rows if r['stereotype_type'] == stype][-2:]
            print("  Examples:")
            for i, r in enumerate(examples):
                print(f"    [{i}] {r['sentence'][:80]}...")

    df = pd.DataFrame(rows)
    if df.empty:
        print("\n[ERROR] No data extracted!")
        return df

    # Build a robust group id:
    # Prefer document_id if present; otherwise fall back to a stable hash of sentence.
    # (This ensures GroupSplit always works.)
    df['group_id'] = df['document_id'].astype(str).fillna('').str.strip()
    missing = df['group_id'].eq('') | df['group_id'].eq('None')
    if missing.any():
        df.loc[missing, 'group_id'] = df.loc[missing, 'sentence'].apply(_stable_hash)

    print(f"\n[SUCCESS] Total extracted samples: {len(df)}")
    print(f"[INFO] Unique groups (document_id/group_id): {df['group_id'].nunique()}")
    return df

def clean_basic(df):
    """Clean without balancing (balancing will be done per split to keep split integrity)."""
    if df.empty:
        print("[ERROR] Empty DataFrame, skipping cleaning")
        return df

    print(f"\n[INFO] Data cleaning...")
    print(f"  Original size: {len(df)}")

    # Remove duplicates by sentence
    original_len = len(df)
    df = df.drop_duplicates(subset=['sentence']).reset_index(drop=True)
    print(f"  After deduplication: {len(df)} (removed {original_len - len(df)})")

    # Remove very short sentences
    df = df[df['sentence'].str.len() >= 10].reset_index(drop=True)
    print(f"  After length filter: {len(df)}")

    return df

def balance_within_split(df_split, seed=42):
    """Balance labels inside a split (keeps pro/anti groups already fixed by group split)."""
    counts = df_split['label'].value_counts()
    if len(counts) < 2:
        return df_split

    n = counts.min()
    df0 = df_split[df_split['label'] == 0].sample(n=n, random_state=seed)
    df1 = df_split[df_split['label'] == 1].sample(n=n, random_state=seed)
    return pd.concat([df0, df1]).sample(frac=1, random_state=seed).reset_index(drop=True)

def group_split(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    groups = df['group_id'].values

    # 1) train vs temp
    gss1 = GroupShuffleSplit(n_splits=1, test_size=(1 - train_ratio), random_state=seed)
    train_idx, temp_idx = next(gss1.split(df, y=df['label'], groups=groups))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)

    # 2) val vs test (split temp groups)
    val_fraction_of_temp = val_ratio / (val_ratio + test_ratio)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=(1 - val_fraction_of_temp), random_state=seed)
    val_idx, test_idx = next(gss2.split(temp_df, y=temp_df['label'], groups=temp_df['group_id'].values))

    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    return train_df, val_df, test_df

def analyze_split(name, df_split):
    counts = df_split['label'].value_counts()
    stereo = counts.get(1, 0)
    non_stereo = counts.get(0, 0)
    print(f"  {name}: rows={len(df_split)}, groups={df_split['group_id'].nunique()}, "
          f"stereotype={stereo}, non-stereotype={non_stereo}")

def split_and_save_grouped(df, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42, balance=True):
    if df.empty:
        print("\n[ERROR] Empty dataset, cannot save")
        return

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[INFO] Group-splitting dataset by document_id/group_id (no leakage)...")
    train_df, val_df, test_df = group_split(df, train_ratio, val_ratio, test_ratio, seed)

    print("\n[INFO] Before balancing:")
    analyze_split("train", train_df)
    analyze_split("valid", val_df)
    analyze_split("test", test_df)

    if balance:
        print("\n[INFO] Balancing labels within each split...")
        train_df = balance_within_split(train_df, seed=seed)
        val_df = balance_within_split(val_df, seed=seed)
        test_df = balance_within_split(test_df, seed=seed)

        print("\n[INFO] After balancing:")
        analyze_split("train", train_df)
        analyze_split("valid", val_df)
        analyze_split("test", test_df)

    # Save CSV
    train_path = os.path.join(output_dir, 'train.csv')
    val_path = os.path.join(output_dir, 'valid.csv')
    test_path = os.path.join(output_dir, 'test.csv')

    ##
    train_df.to_csv(train_path, index=False, encoding='utf-8')
    val_df.to_csv(val_path, index=False, encoding='utf-8')
    test_df.to_csv(test_path, index=False, encoding='utf-8')

    print(f"\n{'='*70}")
    print("[SUCCESS] Files saved!")
    print(f"{'='*70}")
    print(f"\nOutput directory: {output_dir}")
    print(f"  train.csv: {len(train_df)} rows")
    print(f"  valid.csv: {len(val_df)} rows")
    print(f"  test.csv:  {len(test_df)} rows")
    print("\nTip: group_id 保留用于验证：同一个 group 不会同时出现在不同 split。")

def main():
    parser = argparse.ArgumentParser(description="Preprocess WinoBias dataset (group split)")
    parser.add_argument("--output_dir", type=str, default="dataset_winobias_grouped",
                        help="Output directory for processed CSV files")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_balance", action="store_true",
                        help="Disable label balancing within each split")
    args = parser.parse_args()

    print("="*70)
    print("WinoBias Preprocessing (Group Split by document_id)")
    print("="*70)

    datasets = load_winobias_data()
    if not datasets:
        print("\n[ERROR] Failed to load datasets")
        return

    df = process_winobias_samples(datasets)
    if df.empty:
        print("\n[ERROR] Data extraction failed")
        return

    df = clean_basic(df)
    if df.empty:
        print("\n[ERROR] Data is empty after cleaning")
        return

    split_and_save_grouped(
        df,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        balance=(not args.no_balance)
    )

if __name__ == "__main__":
    main()