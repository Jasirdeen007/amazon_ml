#!/usr/bin/env python3
"""
Phase 1: Comprehensive Dataset Analysis for Amazon ML Challenge 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import json

BASE_PATH = Path("d:/amazon_ml_challenge/6ab10eb3b23ba_student_resource/student_resource/dataset")
TRAIN_PATH = BASE_PATH / "train"
TEST_PATH = BASE_PATH / "test"
OUTPUT_PATH = Path("d:/amazon_ml_challenge/data/processed")

def load_data(filepath, nrows=None):
    try:
        return pd.read_csv(filepath, sep="\t", nrows=nrows)
    except Exception as e:
        print(f"Error: {e}")
        return None

def analyze_file(filepath, name):
    print(f"\n{'='*80}\nFILE: {name}\n{'='*80}")
    
    df = load_data(filepath)
    if df is None:
        return None
    
    size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"Size: {size_mb:.2f} MB | Rows: {len(df):,} | Columns: {len(df.columns)}")
    print(f"Columns: {list(df.columns)}")
    print(f"Null values:\n{df.isnull().sum()}")
    print(f"\nFirst record sample:")
    for col in df.columns:
        val = str(df[col].iloc[0])[:80]
        print(f"  {col}: {val}")
    
    return df

def analyze_ground_truth(filepath):
    print(f"\n{'='*80}\nGROUND TRUTH ANALYSIS\n{'='*80}")
    
    df = load_data(filepath)
    if df is None:
        return None
    
    print(f"Total Source 1 Entities: {len(df):,}")
    
    match_counts = []
    for entity_ids in df['matched_entity_ids']:
        if pd.isna(entity_ids) or entity_ids == '':
            match_counts.append(0)
        else:
            match_counts.append(len(str(entity_ids).split(',')))
    
    counter = Counter(match_counts)
    print(f"\nMatch Distribution:")
    print(f"  Singletons (0 matches): {counter[0]:,} ({100*counter[0]/len(df):.1f}%)")
    
    for cnt in sorted([k for k in counter.keys() if k > 0])[:5]:
        print(f"  Entities with {cnt} match(es): {counter[cnt]:,}")
    
    print(f"\nStatistics:")
    print(f"  Average matches per entity: {np.mean(match_counts):.2f}")
    print(f"  Total match relationships: {sum(match_counts):,}")
    
    return df

def analyze_country(filepath, name):
    df = load_data(filepath)
    if df is None or 'country' not in df.columns:
        return
    
    print(f"\n{name}:")
    for country, count in df['country'].value_counts().items():
        pct = 100 * count / len(df)
        print(f"  {country}: {count:,} ({pct:.1f}%)")

def main():
    print("\n" + "="*80)
    print("AMAZON ML CHALLENGE 2026 - PHASE 1: DATA EXPLORATION")
    print("="*80)
    
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    print("\n### TRAINING DATA ###")
    train_s1 = analyze_file(TRAIN_PATH / "train_source1.tsv", "train_source1.tsv")
    train_s2 = analyze_file(TRAIN_PATH / "train_source2.tsv", "train_source2.tsv")
    train_s3 = analyze_file(TRAIN_PATH / "train_source3.tsv", "train_source3.tsv")
    train_gt = analyze_ground_truth(TRAIN_PATH / "train_ground_truth.tsv")
    
    print("\n### COUNTRY DISTRIBUTION (TRAINING) ###")
    analyze_country(TRAIN_PATH / "train_source1.tsv", "Source 1")
    analyze_country(TRAIN_PATH / "train_source2.tsv", "Source 2")
    analyze_country(TRAIN_PATH / "train_source3.tsv", "Source 3")
    
    print("\n### TEST DATA ###")
    test_s1 = analyze_file(TEST_PATH / "test_source1.tsv", "test_source1.tsv")
    test_s2 = analyze_file(TEST_PATH / "test_source2.tsv", "test_source2.tsv")
    test_s3 = analyze_file(TEST_PATH / "test_source3.tsv", "test_source3.tsv")
    
    print("\n### COUNTRY DISTRIBUTION (TEST) ###")
    analyze_country(TEST_PATH / "test_source1.tsv", "Test Source 1")
    analyze_country(TEST_PATH / "test_source2.tsv", "Test Source 2")
    analyze_country(TEST_PATH / "test_source3.tsv", "Test Source 3")
    
    # Summary
    summary = {
        "training": {
            "source1": len(train_s1) if train_s1 is not None else 0,
            "source2": len(train_s2) if train_s2 is not None else 0,
            "source3": len(train_s3) if train_s3 is not None else 0,
            "ground_truth": len(train_gt) if train_gt is not None else 0,
        },
        "test": {
            "source1": len(test_s1) if test_s1 is not None else 0,
            "source2": len(test_s2) if test_s2 is not None else 0,
            "source3": len(test_s3) if test_s3 is not None else 0,
        }
    }
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Training - S1: {summary['training']['source1']:,}, S2: {summary['training']['source2']:,}, S3: {summary['training']['source3']:,}")
    print(f"Test - S1: {summary['test']['source1']:,}, S2: {summary['test']['source2']:,}, S3: {summary['test']['source3']:,}")
    
    with open(OUTPUT_PATH / "data_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nExploration complete. Summary saved to {OUTPUT_PATH / 'data_summary.json'}\n")

if __name__ == "__main__":
    main()
