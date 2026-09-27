#!/usr/bin/env python3
"""Phase 2: Data Normalization"""
import pandas as pd
from pathlib import Path
import re
from unidecode import unidecode

BASE_PATH = Path("d:/amazon_ml_challenge/6ab10eb3b23ba_student_resource/student_resource/dataset")
OUTPUT_PATH = Path("d:/amazon_ml_challenge/data/processed")

def normalize_name(name):
    """Normalize business name"""
    if pd.isna(name):
        return ""
    name = str(name).lower().strip()
    # Remove special chars, standardize suffixes
    replacements = {
        r'\bpvt\.?\s+ltd\.?\b': 'pvt ltd',
        r'\bprivate\s+limited\b': 'pvt ltd',
        r'\binc\.?\b': 'inc',
        r'\bincorporated\b': 'inc',
        r'\bcorp\.?\b': 'corp',
        r'\bcorporation\b': 'corp',
        r'\b&\b': 'and',
    }
    for pattern, replacement in replacements.items():
        name = re.sub(pattern, replacement, name)
    name = re.sub(r'[^a-z0-9\s]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def normalize_address(addr):
    """Normalize address"""
    if pd.isna(addr) or addr == '':
        return ""
    addr = str(addr).lower().strip()
    replacements = {
        r'\brd\.?\b': 'road',
        r'\bst\.?\b': 'street',
        r'\bave\.?\b': 'avenue',
        r'\bno\.?\b': 'no',
    }
    for pattern, replacement in replacements.items():
        addr = re.sub(pattern, replacement, addr)
    addr = unidecode(addr)
    addr = re.sub(r'[^a-z0-9\s,]', '', addr)
    addr = re.sub(r'\s+', ' ', addr).strip()
    return addr

def process_file(filepath, output_name):
    """Process and save normalized data"""
    print(f"Processing {filepath.name}...")
    df = pd.read_csv(filepath, sep="\t")
    
    df['business_name_norm'] = df['business_name'].apply(normalize_name)
    df['business_address_norm'] = df['business_address'].apply(normalize_address)
    
    df.to_csv(OUTPUT_PATH / output_name, sep="\t", index=False)
    print(f"  ✓ {output_name} saved ({len(df):,} rows)")
    return df

# Process training data
train_s1 = process_file(BASE_PATH / "train" / "train_source1.tsv", "train_source1_norm.tsv")
train_s2 = process_file(BASE_PATH / "train" / "train_source2.tsv", "train_source2_norm.tsv")
train_s3 = process_file(BASE_PATH / "train" / "train_source3.tsv", "train_source3_norm.tsv")

# Process test data
test_s1 = process_file(BASE_PATH / "test" / "test_source1.tsv", "test_source1_norm.tsv")
test_s2 = process_file(BASE_PATH / "test" / "test_source2.tsv", "test_source2_norm.tsv")
test_s3 = process_file(BASE_PATH / "test" / "test_source3.tsv", "test_source3_norm.tsv")

print("\n✅ Normalization complete!")
