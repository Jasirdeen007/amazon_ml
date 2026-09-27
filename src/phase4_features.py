#!/usr/bin/env python3
"""Phase 4: Feature Engineering for candidate pairs"""
import pandas as pd
import numpy as np
from pathlib import Path
from rapidfuzz import fuzz

OUTPUT_PATH = Path("d:/amazon_ml_challenge/data/processed")
FEATURES_PATH = Path("d:/amazon_ml_challenge/data/features")
FEATURES_PATH.mkdir(exist_ok=True)

def compute_string_features(name1, name2, addr1, addr2):
    """Compute string similarity features"""
    features = {}
    
    # Name similarity
    features['name_jw'] = fuzz.jaro_winkler(name1, name2)
    features['name_ratio'] = fuzz.ratio(name1, name2) / 100.0
    features['name_partial'] = fuzz.partial_ratio(name1, name2) / 100.0
    
    # Address similarity
    if addr1 and addr2:
        features['addr_jw'] = fuzz.jaro_winkler(addr1, addr2)
        features['addr_ratio'] = fuzz.ratio(addr1, addr2) / 100.0
    else:
        features['addr_jw'] = 0.0
        features['addr_ratio'] = 0.0
    
    # Token overlap
    name1_tokens = set(name1.split())
    name2_tokens = set(name2.split())
    overlap = len(name1_tokens & name2_tokens) / max(len(name1_tokens | name2_tokens), 1)
    features['name_token_overlap'] = overlap
    
    return features

def generate_features(candidates_df, s1, s2, s3):
    """Generate features for all candidate pairs"""
    # Create lookup dicts
    s1_dict = dict(zip(s1['entity_id'], zip(s1['business_name_norm'], s1['business_address_norm'])))
    s2_dict = dict(zip(s2['entity_id'], zip(s2['business_name_norm'], s2['business_address_norm'])))
    s3_dict = dict(zip(s3['entity_id'], zip(s3['business_name_norm'], s3['business_address_norm'])))
    
    combined_dict = {**s2_dict, **s3_dict}
    
    features_list = []
    
    for idx, row in candidates_df.iterrows():
        s1_id = row['source1_entity_id']
        s2_s3_id = row['matched_entity_id']
        
        s1_name, s1_addr = s1_dict.get(s1_id, ('', ''))
        s2_s3_name, s2_s3_addr = combined_dict.get(s2_s3_id, ('', ''))
        
        feat = compute_string_features(s1_name, s2_s3_name, s1_addr, s2_s3_addr)
        feat['source1_entity_id'] = s1_id
        feat['matched_entity_id'] = s2_s3_id
        
        features_list.append(feat)
    
    return pd.DataFrame(features_list)

def main():
    print("Loading candidate pairs...")
    candidates = pd.read_csv(OUTPUT_PATH / "candidate_pairs.tsv", sep="\t")
    
    print("Loading normalized data...")
    s1 = pd.read_csv(OUTPUT_PATH / "train_source1_norm.tsv", sep="\t")
    s2 = pd.read_csv(OUTPUT_PATH / "train_source2_norm.tsv", sep="\t")
    s3 = pd.read_csv(OUTPUT_PATH / "train_source3_norm.tsv", sep="\t")
    
    print(f"Computing features for {len(candidates):,} candidates...")
    features_df = generate_features(candidates, s1, s2, s3)
    
    features_df.to_csv(FEATURES_PATH / "candidate_features.tsv", sep="\t", index=False)
    print(f"✅ Features saved: {len(features_df):,} candidates with features")

if __name__ == "__main__":
    main()
