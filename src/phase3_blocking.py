#!/usr/bin/env python3
"""Phase 3: Multi-Pass Blocking - Generate candidate pairs"""
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import pickle

OUTPUT_PATH = Path("d:/amazon_ml_challenge/data/processed")
CANDIDATES_PATH = Path("d:/amazon_ml_challenge/data/candidates")
CANDIDATES_PATH.mkdir(exist_ok=True)

def load_normalized_data():
    """Load normalized data"""
    s1 = pd.read_csv(OUTPUT_PATH / "train_source1_norm.tsv", sep="\t")
    s2 = pd.read_csv(OUTPUT_PATH / "train_source2_norm.tsv", sep="\t")
    s3 = pd.read_csv(OUTPUT_PATH / "train_source3_norm.tsv", sep="\t")
    return s1, s2, s3

def load_ground_truth():
    """Load ground truth"""
    gt = pd.read_csv(
        Path("d:/amazon_ml_challenge/6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_ground_truth.tsv"),
        sep="\t"
    )
    return gt

def blocking_exact_key(s1, s2, s3):
    """Exact key blocking: normalized name + country"""
    candidates = set()
    
    # Create lookup: (name, country) -> list of entity_ids
    s2_lookup = defaultdict(list)
    s3_lookup = defaultdict(list)
    
    for idx, row in s2.iterrows():
        key = (row['business_name_norm'], row['country'])
        s2_lookup[key].append(row['entity_id'])
    
    for idx, row in s3.iterrows():
        key = (row['business_name_norm'], row['country'])
        s3_lookup[key].append(row['entity_id'])
    
    # For each S1 entity, find candidates with same name+country
    for idx, row in s1.iterrows():
        s1_id = row['entity_id']
        key = (row['business_name_norm'], row['country'])
        
        for s2_id in s2_lookup.get(key, []):
            candidates.add((s1_id, s2_id))
        for s3_id in s3_lookup.get(key, []):
            candidates.add((s1_id, s3_id))
    
    return candidates

def blocking_phonetic_key(s1, s2, s3):
    """Phonetic blocking: first 4 chars of normalized name"""
    candidates = set()
    
    s2_lookup = defaultdict(list)
    s3_lookup = defaultdict(list)
    
    for idx, row in s2.iterrows():
        phonetic_key = row['business_name_norm'][:4] if len(row['business_name_norm']) >= 4 else row['business_name_norm']
        s2_lookup[(phonetic_key, row['country'])].append(row['entity_id'])
    
    for idx, row in s3.iterrows():
        phonetic_key = row['business_name_norm'][:4] if len(row['business_name_norm']) >= 4 else row['business_name_norm']
        s3_lookup[(phonetic_key, row['country'])].append(row['entity_id'])
    
    for idx, row in s1.iterrows():
        s1_id = row['entity_id']
        phonetic_key = row['business_name_norm'][:4] if len(row['business_name_norm']) >= 4 else row['business_name_norm']
        
        for s2_id in s2_lookup.get((phonetic_key, row['country']), []):
            candidates.add((s1_id, s2_id))
        for s3_id in s3_lookup.get((phonetic_key, row['country']), []):
            candidates.add((s1_id, s3_id))
    
    return candidates

def measure_blocking_recall(candidates, gt):
    """Measure blocking recall"""
    true_matches = set()
    for idx, row in gt.iterrows():
        s1_id = row['source1_entity_id']
        if pd.isna(row['matched_entity_ids']) or row['matched_entity_ids'] == '':
            continue
        for s2_s3_id in str(row['matched_entity_ids']).split(','):
            s2_s3_id = s2_s3_id.strip()
            true_matches.add((s1_id, s2_s3_id))
    
    recall = len(true_matches & candidates) / len(true_matches) if len(true_matches) > 0 else 0
    compression = len(candidates) / (2.2e6 * 5e6) if len(candidates) > 0 else 0
    
    return recall, compression, len(candidates)

def main():
    print("Loading data...")
    s1, s2, s3 = load_normalized_data()
    gt = load_ground_truth()
    
    print("Blocking Pass 1: Exact Key...")
    candidates_exact = blocking_exact_key(s1, s2, s3)
    print(f"  Candidates: {len(candidates_exact):,}")
    
    print("Blocking Pass 2: Phonetic Key...")
    candidates_phonetic = blocking_phonetic_key(s1, s2, s3)
    print(f"  Candidates: {len(candidates_phonetic):,}")
    
    # Union
    all_candidates = candidates_exact | candidates_phonetic
    print(f"\nTotal Candidates (Union): {len(all_candidates):,}")
    
    # Measure recall
    recall, compression, count = measure_blocking_recall(all_candidates, gt)
    print(f"Blocking Recall: {recall*100:.2f}%")
    print(f"Candidate Compression: {compression*100:.4f}%")
    
    # Save candidates
    candidate_pairs = pd.DataFrame(list(all_candidates), columns=['source1_entity_id', 'matched_entity_id'])
    candidate_pairs.to_csv(OUTPUT_PATH / "candidate_pairs.tsv", sep="\t", index=False)
    print(f"\n✅ Candidate pairs saved to candidate_pairs.tsv")

if __name__ == "__main__":
    main()
