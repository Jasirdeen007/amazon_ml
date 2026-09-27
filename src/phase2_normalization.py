#!/usr/bin/env python3
"""Phase 2: Data Normalization

This script normalizes raw dataset files from the Amazon ML Challenge.
Teammates must have the raw dataset extracted to:
  6ab10eb3b23ba_student_resource/student_resource/dataset/

Usage:
  python src/phase2_normalization.py
"""
import pandas as pd
from pathlib import Path
import re
import sys
from unidecode import unidecode

# Get project root (parent of 'src' directory)
PROJECT_ROOT = Path(__file__).parent.parent
BASE_PATH = PROJECT_ROOT / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed"

# Validate paths exist
if not BASE_PATH.exists():
    print(f"❌ ERROR: Raw dataset not found at:")
    print(f"   {BASE_PATH.absolute()}")
    print(f"\n📝 SETUP REQUIRED:")
    print(f"   1. Extract the raw dataset ZIP to: 6ab10eb3b23ba_student_resource/")
    print(f"   2. Structure should be:")
    print(f"      amazon_ml/")
    print(f"      ├── 6ab10eb3b23ba_student_resource/")
    print(f"      │   └── student_resource/")
    print(f"      │       └── dataset/")
    print(f"      │           ├── train/")
    print(f"      │           └── test/")
    sys.exit(1)

# Create output directory if it doesn't exist
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

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
    print(f"  Processing {filepath.name}...")
    df = pd.read_csv(filepath, sep="\t")
    
    df['business_name_norm'] = df['business_name'].apply(normalize_name)
    df['business_address_norm'] = df['business_address'].apply(normalize_address)
    
    output_file = OUTPUT_PATH / output_name
    df.to_csv(output_file, sep="\t", index=False)
    
    # Calculate file size
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"    ✓ Saved: {output_name} ({len(df):,} rows | {file_size_mb:.2f} MB)")
    return df

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("PHASE 2: DATA NORMALIZATION")
    print("=" * 80)
    print(f"\nInput:  {BASE_PATH.absolute()}")
    print(f"Output: {OUTPUT_PATH.absolute()}\n")
    
    try:
        # Process training data
        print("[TRAIN DATA]")
        train_s1 = process_file(BASE_PATH / "train" / "train_source1.tsv", "train_source1_norm.tsv")
        train_s2 = process_file(BASE_PATH / "train" / "train_source2.tsv", "train_source2_norm.tsv")
        train_s3 = process_file(BASE_PATH / "train" / "train_source3.tsv", "train_source3_norm.tsv")
        
        # Process test data
        print("\n[TEST DATA]")
        test_s1 = process_file(BASE_PATH / "test" / "test_source1.tsv", "test_source1_norm.tsv")
        test_s2 = process_file(BASE_PATH / "test" / "test_source2.tsv", "test_source2_norm.tsv")
        test_s3 = process_file(BASE_PATH / "test" / "test_source3.tsv", "test_source3_norm.tsv")
        
        # Summary statistics
        total_rows = len(train_s1) + len(train_s2) + len(train_s3) + len(test_s1) + len(test_s2) + len(test_s3)
        total_size_mb = sum((OUTPUT_PATH / f).stat().st_size / (1024 * 1024) 
                           for f in ["train_source1_norm.tsv", "train_source2_norm.tsv", "train_source3_norm.tsv",
                                    "test_source1_norm.tsv", "test_source2_norm.tsv", "test_source3_norm.tsv"])
        
        print("\n" + "=" * 80)
        print("✅ NORMALIZATION COMPLETE!")
        print("=" * 80)
        print(f"\nStatistics:")
        print(f"  Total rows processed: {total_rows:,}")
        print(f"  Total output size: {total_size_mb:.2f} MB")
        print(f"\nAll normalized files saved to:")
        print(f"  {OUTPUT_PATH.absolute()}")
        print(f"\n📝 Next steps:")
        print(f"  1. Verify data/processed/ has all 6 TSV files")
        print(f"  2. Run Phase 3 blocking:")
        print(f"     python src/phase3_blocking_optimized.py")
        print(f"\n")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: File not found: {e}")
        print(f"\nMake sure raw dataset is extracted to: {BASE_PATH.absolute()}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR during normalization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
