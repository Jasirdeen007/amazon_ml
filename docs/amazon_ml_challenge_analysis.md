# Amazon ML Challenge 2026 - Comprehensive Analysis Report

*Last Updated: September 25, 2026*  
*Report Location: `docs/amazon_ml_challenge_analysis.md`*

## Executive Summary

The Amazon ML Challenge 2026 focuses on Business Entity Resolution across three independent data sources with noisy and inconsistent business records. The challenge requires matching Source 2 and Source 3 records to deduplicated Source 1 entities, with evaluation based on F_0.5 score that heavily weights precision over recall. The dataset includes training data from US and India, with test data containing an additional unseen country (France), requiring zero-shot generalization.

---

## 1. Dataset Analysis

### 1.1 Dataset Overview

**File Structure:**
- **Format:** Tab-separated values (.tsv) - must use `sep='\t'` when reading
- **Training Set:** US and India businesses only
- **Test Set:** Additional country (France) not present in training
- **Total Records:** ~2.2 million in Source 1 training data

**File Sizes:**
| File | Size | Description |
|------|------|-------------|
| `train_source1.tsv` | ~200 MB | Source 1 training records (reference) |
| `train_source2.tsv` | ~466 MB | Source 2 training records |
| `train_source3.tsv` | ~480 MB | Source 3 training records |
| `test_source1.tsv` | ~166 MB | Source 1 test records |
| `test_source2.tsv` | ~485 MB | Source 2 test records |
| `test_source3.tsv` | ~482 MB | Source 3 test records |
| `train_ground_truth.tsv` | ~121 MB | Ground truth matches |

### 1.2 Data Schema

Each source file contains these columns:
- **entity_id:** Unique identifier with source prefix (`S1-`, `S2-`, `S3-`)
- **business_name:** Business name with variations and noise
- **business_address:** Address with format inconsistencies
- **country:** Country label (US, India in training; +France in test)

Ground truth file (`train_ground_truth.tsv`):
- **source1_entity_id:** Source 1 entity identifier
- **matched_entity_ids:** Comma-separated list of matching IDs from Sources 2 & 3 (empty for singletons)

### 1.3 Data Quality & Noise Patterns

**Business Name Variations:**
- Abbreviations: Corp vs Corporation, Pvt vs Private, Ltd vs Limited
- Legal suffix inconsistencies
- Punctuation differences (& vs "and")
- Word-order transpositions
- Typos and spelling variations
- Transliteration differences (especially for India)

**Address Variations:**
- Abbreviations: Rd vs Road, St vs Street, Ave vs Avenue
- Missing components: No PIN/ZIP codes, incomplete addresses
- Landmark-based references: "Near SBI ATM", "Behind Post Office"
- Component reordering: Different address format conventions
- Municipal numbering format variations
- Transliteration variants for Indian addresses

**Country-Specific Patterns:**
- **US Addresses:** ZIP codes, state abbreviations, standardized formats
- **India Addresses:** PIN codes, landmark references, complex formatting
- **France Addresses:** Unknown format (zero-shot requirement)

### 1.4 Ground Truth Analysis

- Each Source 1 entity may match **0, 1, or many** records from Sources 2 & 3
- **Singletons:** Entities with no matches - correctly predicting "no match" earns full credit
- Match distribution analysis needed to understand class imbalance

---

## 2. Competition Statement Definition

### 2.1 Problem Statement

**Business Entity Resolution Challenge:** In large-scale commercial platforms, business identity data arrives from multiple independent sources — each contributing partial, noisy fragments of information about the same real-world entities. These fragments share no common identifiers, and the challenge is to determine which records refer to the same business.

---

## 3. Architecture Plans

### 3.1 Architecture 1: Two-Stage Blocking + Classification (Recommended)

**Stage 1: Multi-Key Blocking Strategy**
```
Blocking Keys:
1. Country-level blocking
2. Phonetic encoding (Soundex, Double Metaphone)
3. Address components (PIN/ZIP codes, cities)
4. TF-IDF similarity for name variations
5. Hybrid blocking with multiple passes
```

**Stage 2: Feature Engineering & Classification**
```
Feature Categories:
1. String Similarity Features:
   - Jaccard similarity
   - Levenshtein/edit distance
   - Cosine similarity (TF-IDF)
   - Longest common substring
   
2. Phonetic Matching:
   - Soundex match score
   - Double Metaphone match
   - Phonetic encoding comparison
   
3. Address Features:
   - Component overlap
   - PIN/ZIP code match
   - City name similarity
   - Token-based similarity
   
4. Country-Specific Features:
   - US address format patterns
   - India address patterns
   - Country-specific heuristics

Model: XGBoost/LightGBM with F_0.5 optimization
Threshold: Tuned to maximize F_0.5 on validation set
```

**Advantages:**
- Computationally efficient for large datasets
- Interpretable feature importance
- Easy to optimize for F_0.5 metric
- Good handling of country-specific patterns
- Modular design for experimentation

### 3.2 Architecture 2: Transformer-Based End-to-End

**Approach:**
1. **Text Representation Learning:**
   - Fine-tune BERT/RoBERTa on entity resolution task
   - Combine business_name + business_address + country
   - Learn contextual representations

2. **Cross-encoder Architecture:**
   - Concatenate Source 1 record with candidate record
   - Output similarity score directly
   - Train with contrastive or binary classification loss

3. **Efficient Inference:**
   - Generate embeddings for all records
   - FAISS for approximate nearest neighbor search
   - Hierarchical matching strategy

4. **Zero-Shot Handling:**
   - Leverage pre-trained language model knowledge
   - Generalize to unseen country patterns

**Advantages:**
- Captures semantic similarities beyond string matching
- Better for zero-shot generalization
- Handles complex variations naturally
- Can learn subtle patterns in data

**Challenges:**
- Higher computational requirements
- Less interpretable than traditional methods
- Need efficient inference strategy for large dataset
- Potential overfitting to training country patterns
### 3.4 Hybrid Architecture

**Combined Approach:**
1. **First Pass:** Traditional blocking + classification for efficiency
2. **Second Pass:** Transformer-based verification for ambiguous cases
3. **Third Pass:** Graph-based consistency checking
4. **Ensemble:** Weighted combination of different approaches

---

## 4. Key Recommendations & Implementation Strategy

### 4.1 Priority Areas

1. **Blocking Strategy (40% effort):**
   - Determines recall upper bound
   - Multi-level approach: country → city → phonetic → TF-IDF
   - Conservative blocking to ensure true matches aren't lost

2. **Country-Specific Handling (25% effort):**
   - Separate feature engineering pipelines
   - Address parsing by country format
   - Zero-shot preparation for France

3. **Precision Optimization (20% effort):**
   - F_0.5 focused validation
   - Threshold tuning for precision-recall tradeoff
   - Conservative matching to avoid false merges

4. **Singleton Identification (15% effort):**
   - Confidence thresholds for "no match" decisions
   - Separate classification for singleton detection
   - Validation on singleton performance

### 4.2 Technical Implementation

**Feature Engineering:**
```
# Core features to implement
1. String Similarity:
   - Jaccard similarity (n-gram based)
   - Levenshtein distance (edit distance)
   - Cosine similarity (TF-IDF vectors)
   - Longest common subsequence
   
2. Phonetic Features:
   - Soundex encoding match
   - Double Metaphone comparison
   - Phonetic distance metrics
   
3. Address Features:
   - Component extraction (PIN/ZIP, city, street)
   - Token overlap scores
   - Format pattern matching
   
4. Country-Specific:
   - US: ZIP code validation, state abbreviation mapping
   - India: PIN code validation, landmark recognition
   - Generic: Common address component patterns
```

**Model Selection Criteria:**
1. **XGBoost/LightGBM:** Good baseline, interpretable, efficient
2. **CatBoost:** Handles categorical features well
3. **Siamese Networks:** For embedding-based matching
4. **Transformer Models:** For semantic matching (if compute allows)

### 4.3 Validation Strategy

**Validation Split:**
- Country-stratified split (maintain US/India proportions)
- Time-based split if temporal patterns exist
- Ensure singleton representation in validation

**Metrics to Monitor:**
1. **Primary:** F_0.5 score (macro average)
2. **Secondary:** Precision, Recall, F1-score
3. **Per-country:** Separate scores for US, India
4. **Singleton accuracy:** Correct "no match" predictions

**Cross-validation:**
- 5-fold stratified cross-validation
- Monitor overfitting to country patterns
- Test generalization capability

### 4.4 Computational Considerations

---

## 5. Development Roadmap

### Week 1: Foundation & Exploration
- [ ] Dataset exploration and analysis
- [ ] Basic blocking implementation (country + phonetic)
- [ ] Simple baseline model (string matching)
- [ ] Validation framework setup
- [ ] Initial F_0.5 baseline score

### Week 2: Feature Engineering
- [ ] Advanced string similarity features
- [ ] Phonetic matching implementation
- [ ] Address parsing and normalization
- [ ] Country-specific feature engineering
- [ ] Feature selection and importance analysis

### Week 3: Model Development
- [ ] Model selection and experimentation
- [ ] F_0.5 optimization strategies
- [ ] Threshold tuning for precision-recall
- [ ] Ensemble method exploration
- [ ] Cross-validation and hyperparameter tuning

### Week 4: Refinement & Submission
- [ ] Zero-shot generalization testing
- [ ] Final model training on full dataset
- [ ] Submission format validation
- [ ] Documentation preparation
- [ ] Final submission package assembly

---

## 6. Success Criteria & Evaluation

### Minimum Viable Solution
- F_0.5 score > 0.6 on validation set
- Handles all three countries (US, India, France)
- Computationally feasible on provided hardware
- Follows all competition rules and constraints

### Target Solution
- F_0.5 score > 0.75 on validation set
- Robust generalization to unseen country
- Efficient processing of full dataset
- Well-documented and reproducible

### Stretch Goals
- F_0.5 score > 0.85
- Novel approach or innovation
- Production-ready code quality
- Comprehensive error analysis

---

## 7. Resources & References

### Technical Resources
- **String Matching:** Jaccard, Levenshtein, TF-IDF algorithms
- **Phonetic Encoding:** Soundex, Metaphone, Double Metaphone
- **Machine Learning:** XGBoost, LightGBM, scikit-learn
- **Deep Learning:** Transformers, Siamese networks
- **Efficient Search:** FAISS, approximate nearest neighbors

### Competition Resources
- **Official Documentation:** README.md in student_resource
- **Submission Template:** Documentation_template.md
- **Validation Script:** utils/validate_submission.py
- **Dataset:** Provided TSV files in dataset/ directory

### Key Considerations
1. **No External Data:** Strict prohibition on data lookup
2. **Zero-Shot Learning:** Must handle France without training examples
3. **Precision Focus:** F_0.5 metric prioritizes avoiding false merges
4. **Scalability:** Large dataset requires efficient algorithms

---

*Report generated: September 25, 2026*  
*Analysis based on Amazon ML Challenge 2026 dataset and documentation*
**Efficiency Strategies:**
1. **Vectorized Operations:** Use numpy/pandas vectorization
2. **Approximate Methods:** FAISS for similarity search
3. **Parallel Processing:** Multiprocessing for feature computation
4. **Memory Management:** Chunk processing for large files
5. **Caching:** Cache intermediate results

**Scalability Planning:**
- Start with subset for experimentation
- Scale gradually to full dataset
- Monitor memory usage and runtime
- Consider distributed computing if needed

### 4.5 Risk Mitigation

**High-Risk Areas:**
1. **Overfitting to training countries:** May fail on France
2. **Poor blocking strategy:** Misses true matches irrecoverably
3. **Precision-recall imbalance:** Wrong optimization target
4. **Computational bottlenecks:** Can't process full dataset

**Mitigation Strategies:**
1. **Country-agnostic features:** Focus on generalizable patterns
2. **Conservative blocking:** Multiple blocking strategies with union
3. **F_0.5 focused optimization:** Regular validation on target metric
4. **Efficient implementation:** Profiling and optimization early

### 4.6 Submission Requirements

**Final Submission Package Must Include:**
1. **`output/matching_results.tsv`:** Final matches (leaderboard)
2. **`output/candidate_pairs.tsv`:** Candidate generation results
3. **`code/business_entity_resolution/`:** Runnable pipeline
   - `src/` with all source code
   - `README.md` with setup instructions
   - `requirements.txt` with dependencies
4. **Methodology document:** Following provided template

**Format Requirements:**
- Tab-separated files (.tsv)
- Exact column names as specified
- Proper handling of empty matches for singletons
- All test entities must appear in submission

### 3.3 Architecture 3: Graph-Based Clustering

**Stage 1: Similarity Graph Construction**
```
1. Compute pairwise similarity for candidate pairs
2. Build weighted graph:
   - Nodes: All business records
   - Edges: Similarity scores above threshold
   - Weights: Similarity confidence
   
3. Graph features:
   - Node degree analysis
   - Community structure
   - Connectivity patterns
```

**Stage 2: Community Detection**
```
1. Connected components for high-confidence matches
2. Louvain/Leiden community detection
3. Hierarchical clustering with multiple thresholds
4. Cluster validation and refinement
```

**Stage 3: Post-processing**
```
1. Ensure Source 1 entities anchor clusters
2. Resolve contradictory matches
3. Apply confidence thresholds
4. Handle singletons explicitly
```

**Advantages:**
- Naturally handles many-to-many relationships
- Captures transitive similarities (A similar to B, B similar to C → A similar to C)
- Good for complex matching patterns
- Can identify cluster-level patterns

**Challenges:**
- Scalability for large datasets (2.2M+ nodes)
- Parameter tuning complexity
- Need for efficient graph algorithms
- Memory requirements for similarity matrix
### 2.2 Task Definition

Given business records from 3 independent data sources with noisy and inconsistent fields, determine which records across sources refer to the same real-world business entity.

**Key Details:**
- Source 1 is the deduplicated reference source
- Task: Find all matching records from Source 2 and Source 3 for each Source 1 entity
- A Source 1 entity may match zero, one, or many records from Source 2 and Source 3

### 2.3 Evaluation Metric

**F_0.5 Score (macro average):**
- Precision-weighted metric (2× weight on precision vs recall)
- Formula: `F_0.5 = (1.25 × precision × recall) / (0.25 × precision + recall)`

**Why precision-heavy?**
- In real-world entity resolution, merging two distinct businesses (false positive) is more damaging than missing a link (false negative)
- F_0.5 weights precision 2× over recall
- Singleton entities contribute 1.0 score when correctly predicted as no-match

### 2.4 Constraints & Limitations

**STRICTLY PROHIBITED:**
- External data lookup (APIs, databases, geocoding services)
- Commercial entity resolution services
- Government database lookups
- Any external data augmentation

**Requirements:**
- Must handle zero-shot generalization to new country (France)
- Must use only provided training data
- All test entities (including France) must appear in submissions