# Memory.md - Amazon ML Challenge 2026 Progress Tracker

## Project Overview
- **Challenge:** Business Entity Resolution across 3 data sources
- **Goal:** Match Source 2 & Source 3 records to Source 1 (reference) entities
- **Metric:** F0.5 (precision-weighted)
- **GPU:** RTX 2050 with 4GB VRAM
- **Python:** 3.14.3

## Important Checkpoints

### Phase 1: Data Exploration & Setup ✅ COMPLETE

#### 1.1 Dataset Structure (VERIFIED)
- **Training Data:** ~1.3 GB total
  - train_source1.tsv: 200 MB → **2,206,821 records**
  - train_source2.tsv: 467 MB → **5,034,616 records**
  - train_source3.tsv: 480 MB → **5,285,603 records**
  - train_ground_truth.tsv: 127 MB

- **Test Data:** ~1.2 GB total
  - test_source1.tsv: 167 MB → **1,732,544 records**
  - test_source2.tsv: 486 MB → **4,887,273 records**
  - test_source3.tsv: 483 MB → **5,082,316 records**
  - **⚠️ CRITICAL: Test data includes FRANCE (15% of test set) - unseen in training!**

#### 1.2 Schema Confirmed (All Source Files)
- `entity_id`: Unique ID with source prefix (S1-, S2-, S3-)
- `business_name`: Some null values (S2: 2, S3: 13-59)
- `business_address`: **Significant nulls!** S2: 169K (3.4%), S3: 176K (3.3%)
- `country`: No nulls, values are US, India, France (test only)

#### 1.3 Ground Truth Analysis (CRITICAL INSIGHTS)
- **Total Source 1 Entities:** 2,206,821
- **Match Distribution:**
  - Singletons (0 matches): 123,247 (5.6%)
  - 1 match: 119,157 (5.4%)
  - 2 matches: 375,212 (17.0%)
  - 3 matches: 530,841 (24.0%)
  - 4 matches: 484,115 (21.9%)
  - 5+ matches: 574,248 (26.0%)
  
- **Statistics:**
  - Average matches per entity: **3.46**
  - Total match relationships: **7,638,365**
  - **Key:** 94.4% are non-singletons (multi-match problem!)

#### 1.4 Country Distribution (CRITICAL FOR GENERALIZATION)
- **Training:** 60% US, 40% India
- **Test:** 38.3% US, 46.8% India, 15% France (NEW!)
- **⚠️ Critical Issue:** France is unseen in training - zero-shot generalization required!

#### 1.5 Data Quality Issues
1. **Missing business_name:** Very rare (2-59 records per source)
2. **Missing business_address:** 3-3.4% of S2/S3 records have empty addresses
3. **Address format variations:** Partial addresses, landmarks, multi-component
4. **Text variations:** Hindi script, abbreviations, transliterations

### Phase 1 Setup Status ✅ COMPLETE
✅ Virtual environment created at: d:\amazon_ml_challenge\venv
✅ Dependencies installed successfully:
  - pandas 3.0.6, numpy 2.5.3, scikit-learn 1.9.1
  - xgboost 3.4.1, lightgbm 4.7.0
  - rapidfuzz 3.14.6, fuzzywuzzy 0.18.0
  - nltk 3.10.3, unidecode 1.4.0

✅ Project structure created with all directories
✅ Phase 1 data exploration script completed and executed
✅ Data summary saved to: d:\amazon_ml_challenge\data\processed\data_summary.json

### Technical Decisions Made
1. **Not using PyTorch immediately** - Will start with XGBoost/LightGBM for baseline
2. **TensorFlow excluded** - PyTorch is lighter for our GPU constraints
3. **Multi-pass blocking** strategy planned: Exact Keys, Address Keys, Phonetic Keys, TF-IDF
4. **Focusing on precision** due to F0.5 metric (2x weight on precision)

### Important Notes
- **DO NOT** use external APIs, geocoding services, or government databases
- All output must be reproducible from code
- candidate_pairs.tsv is part of final submission (blocking quality is evaluated)
- Ground truth may have many-to-many relationships
- Singletons (no matches) contribute 1.0 to F0.5 when correctly predicted

### Directory Structure (Planned)
```
d:\amazon_ml_challenge\
├── venv/                          # Virtual environment
├── requirements.txt               # Dependencies
├── memory.md                      # This file
├── data/                          # Working data directory
│   ├── processed/                # Normalized data
│   ├── features/                 # Feature engineering outputs
│   └── candidates/               # Candidate pairs
├── src/                          # Source code
│   ├── data_loader.py
│   ├── normalizer.py
│   ├── blocking.py
│   ├── features.py
│   ├── model.py
│   └── pipeline.py
├── notebooks/                    # Exploratory notebooks


### Phase 2 Plan (NEXT - DATA NORMALIZATION & BLOCKING)
1. **Data Normalization**
   - Handle missing addresses (~3% of records)
   - Normalize business names
   - Normalize addresses (country-agnostic)
   - Transliteration handling for Hindi

2. **Blocking System**
   - Multi-pass blocking (exact, address, phonetic, TF-IDF)
   - Generate candidate_pairs.tsv
   - Measure blocking recall (target: >99%)

3. **Feature Engineering**
   - String similarity metrics
   - Phonetic features
   - Address features
   - Country-aware patterns

4. **Model Training**
   - XGBoost/LightGBM
   - F0.5 optimization
   - Singleton handling

### Lessons Learned / Corrections from Reference Docs
- **VERIFIED:** All files are TSV format with `sep="\t"`
- **VERIFIED:** France (15%) IS in test set - critical for zero-shot generalization
- **CORRECTED:** Missing addresses worse than expected (~3% completely empty, many partial)

---

**Last Updated:** 2026-09-26 17:58
**Current Phase:** 1 - Complete ✅
**Next Phase:** 2 - Data Normalization & Blocking
**Phase 1 Achievements:**
  - ✅ Environment setup
  - ✅ Dependencies installed
  - ✅ Data exploration complete
  - ✅ Summary statistics documented
├── output/                       # Final submissions
│   ├── candidate_pairs.tsv
│   └── matching_results.tsv
└── docs/                         # Documentation
    ├── DECISIONS.md
    ├── EXPERIMENT_LOG.md
    └── PROJECT_STATE.md
```

### Lessons Learned / Corrections

*To be updated as we discover issues with reference docs or implementations*

---

**Last Updated:** 2026-09-26
**Current Phase:** 1.5 - Dependencies Installation
**Estimated Phase 1 Completion:** After initial data exploration and baseline metrics
