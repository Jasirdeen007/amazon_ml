# Amazon ML Challenge 2026 — Business Entity Resolution Plan

**Last Updated:** September 25, 2026

## 1. Executive Summary

The Amazon ML Challenge 2026 is a **Business Entity Resolution** problem.

We have three independent sources of business records:

- **Source 1:** deduplicated/reference entities
- **Source 2:** noisy business records
- **Source 3:** noisy business records

The goal is to determine which Source 2 and Source 3 records refer to each Source 1 business.

The main difficulty is that the same business can appear with:

- different spellings
- abbreviations
- reordered words
- incomplete addresses
- typos
- transliteration differences
- different address formats

The training data contains **US and India**, while the test data additionally contains **France**, creating a **zero-shot generalization** requirement.

The proposed primary solution is:

> **Normalize → Multi-pass Blocking → Pairwise Feature Engineering → XGBoost/LightGBM → Conservative Thresholding → Optional Verification/Graph Consistency**

This is feasible, but the most important part is **candidate generation (blocking)**. We must avoid comparing every business against every other business.

---

# 2. What We Are Actually Building

For every Source 1 business, the pipeline will:

```text
Source 1 business
      |
      v
Generate a small set of possible candidates
from Source 2 + Source 3
      |
      v
Calculate similarity features
for each candidate pair
      |
      v
XGBoost / LightGBM
      |
      v
Match probability
      |
      v
Conservative threshold
      |
      +----> Match
      |
      +----> No Match
```

Example:

```text
S1:
"ABC Technologies Pvt Ltd"
"12 MG Road, Bangalore 560001"

Possible candidates:

S2:
"ABC Technology Private Limited"
"12, Mahatma Gandhi Road, Bengaluru - 560001"

S3:
"ABC Tech Pvt. Ltd."
"12 MG Rd Bangalore 560001"

The model should recognize that these records
probably represent the same business.
```

The model is **not directly predicting the business name**.

It is predicting:

> **Does this pair of records represent the same real-world business?**

That distinction is important.

---

# 3. Dataset

## 3.1 File Structure

The data is provided as tab-separated files (`.tsv`).

When loading:

```python
pd.read_csv(path, sep="\t")
```

Training data contains US and India businesses.

Test data additionally contains France.

Approximate Source 1 training size:

- **~2.2 million records**

Approximate file sizes:

| File | Size | Purpose |
|---|---:|---|
| `train_source1.tsv` | ~200 MB | Reference entities |
| `train_source2.tsv` | ~466 MB | Source 2 |
| `train_source3.tsv` | ~480 MB | Source 3 |
| `test_source1.tsv` | ~166 MB | Test reference entities |
| `test_source2.tsv` | ~485 MB | Test Source 2 |
| `test_source3.tsv` | ~482 MB | Test Source 3 |
| `train_ground_truth.tsv` | ~121 MB | Known training matches |

---

# 4. Dataset Features

Each source contains:

| Feature | Meaning | Why it matters |
|---|---|---|
| `entity_id` | Unique record identifier | Identifies the record |
| `business_name` | Business/company name | Strong matching signal |
| `business_address` | Business address | Strong matching signal |
| `country` | Country | Useful for blocking and normalization |

Ground truth contains:

| Column | Meaning |
|---|---|
| `source1_entity_id` | Source 1 reference entity |
| `matched_entity_ids` | Matching Source 2/3 entity IDs |

A Source 1 entity can have:

- **0 matches** → singleton
- **1 match**
- **many matches**

---

# 5. Data Noise We Must Handle

## Business Name

Typical variations:

- `Corp` ↔ `Corporation`
- `Pvt` ↔ `Private`
- `Ltd` ↔ `Limited`
- punctuation differences
- `&` ↔ `and`
- word-order changes
- typos
- spelling variations
- transliteration differences

Example:

```text
ABC Pvt Ltd
ABC Private Limited
A.B.C. Pvt. Ltd.
ABC Technologies Limited
```

## Address

Typical variations:

- `Rd` ↔ `Road`
- `St` ↔ `Street`
- `Ave` ↔ `Avenue`
- missing PIN/ZIP
- incomplete addresses
- landmark references
- different component order
- municipal numbering differences
- transliteration

Example:

```text
12 MG Road, Bangalore 560001
12 Mahatma Gandhi Rd, Bengaluru - 560001
```

## Country Patterns

### US

Common signals:

- ZIP code
- state abbreviation
- standardized address structure

### India

Common signals:

- PIN code
- city/state
- landmarks
- more variable address structure

### France

France is unseen during training.

Therefore we should avoid designing a model that depends heavily on US/India-only rules.

---

# 6. Why We Cannot Compare Everything

Suppose Source 1 has millions of records and Source 2/3 also have millions.

A naive approach would do:

```text
Every Source 1 record
        ×
Every Source 2/3 record
```

This creates an enormous number of comparisons.

Most comparisons are obviously wrong.

For example:

```text
S1: ABC Technologies, Bangalore

There is no reason to compare it with:

XYZ Restaurant, New York
Paris Bakery, Paris
Random Hospital, Mumbai
...
```

We therefore need **blocking**.

---

# 7. What Is Blocking?

**Blocking = quickly selecting only plausible candidates before expensive matching.**

Instead of asking:

> "Which of the millions of records matches this business?"

we ask:

> "Which small subset of records could possibly match this business?"

Example:

```text
2,000,000 records
       |
       | Blocking
       v
2,000–10,000 candidates
       |
       | Similarity features
       v
100–500 strong candidates
       |
       | ML classifier
       v
Final matches
```

The exact candidate count will depend on experiments.

## Important Rule

Blocking should be **high recall**.

If the real matching record is removed during blocking, the ML model can never recover it.

Therefore:

> **Blocking controls the maximum possible recall.**

---

# 8. What Does "Two-Stage Blocking" Mean?

The term can be confusing.

In this plan, **two-stage** means:

### Stage 1 — Candidate Generation

Use cheap rules to find possible matches.

```text
Source 1
   |
   +-- Country
   +-- Address/PIN/ZIP
   +-- City
   +-- Phonetic name
   +-- TF-IDF/name tokens
   |
   v
Candidate pairs
```

### Stage 2 — Candidate Classification

For each candidate pair:

```text
S1 record + candidate record
        |
        v
Similarity features
        |
        v
XGBoost / LightGBM
        |
        v
Match probability
```

So:

> **Blocking answers "Who should we compare?"**

> **Classification answers "Are these two records actually the same business?"**

---

# 9. Multi-Pass Blocking

We should **not depend on one blocking rule**.

For example, this rule:

```text
same country + same city
```

could miss a true match if the city is missing or spelled differently.

Instead, use multiple blocking passes.

### Pass 1 — Country

```text
US ↔ US
India ↔ India
France ↔ France
```

This is a cheap first filter.

### Pass 2 — Address Components

Use:

- PIN/ZIP
- city
- state
- street tokens

### Pass 3 — Phonetic Name

Use:

- Soundex
- Double Metaphone

Useful when names sound similar despite spelling differences.

### Pass 4 — Name Tokens / TF-IDF

Retrieve records with similar business-name tokens.

### Final Candidate Set

Take the **union** of candidates from multiple blocking strategies.

```text
Candidates =
    Country candidates
    UNION
    Address candidates
    UNION
    Phonetic candidates
    UNION
    Name/TF-IDF candidates
```

This is safer than requiring every candidate to satisfy every rule.

---

# 10. Candidate Generation vs Classification

This distinction is central to the project.

## Candidate Generation

Goal:

> Don't miss the true match.

Therefore optimize for **high recall**.

It is acceptable to generate some false candidates.

## Classification

Goal:

> Remove false matches.

Therefore optimize the final **precision/F0.5 tradeoff**.

This stage can be conservative because F0.5 emphasizes precision.

---

# 11. Feature Engineering

Once we have a candidate pair:

```text
Source 1 record
        +
Source 2/3 candidate
```

we create numerical features describing how similar they are.

---

## 11.1 Name Similarity Features

### Levenshtein Distance

Measures the minimum number of character edits needed to transform one string into another.

Example:

```text
ABC LTD
ABC LIMITED
```

Fewer edits → more similar.

Usually convert distance into a normalized similarity score.

### Jaccard Similarity

Compares token overlap.

Example:

```text
A = {"abc", "technology", "pvt"}
B = {"abc", "technology", "limited"}
```

Jaccard:

```text
intersection / union
```

Useful when word order changes.

### TF-IDF Cosine Similarity

Represents text as weighted vectors and measures their angle similarity.

Useful for:

```text
ABC Technology Private Limited
ABC Technologies Pvt Ltd
```

It captures overlapping informative tokens.

### Longest Common Subsequence / Common Substring

Measures how much sequence is shared between two strings.

Useful as an additional character-level signal.

---

# 12. Phonetic Features

Phonetic encoding converts words into codes based approximately on pronunciation.

Examples:

- Soundex
- Metaphone
- Double Metaphone

Useful for:

```text
Stephen
Steven
```

or spelling variations where pronunciation is similar.

For business names, phonetic features should be **supporting signals**, not the only matching rule.

---

# 13. Address Features

Instead of treating the entire address as one string, extract components where possible:

```text
PIN/ZIP
City
State
Street
House/building number
Landmark tokens
```

Then create features such as:

- PIN/ZIP exact match
- city similarity
- state match
- street-token overlap
- address token Jaccard
- normalized address similarity
- house-number match

Example:

```text
Address A:
12 MG Road, Bangalore 560001

Address B:
12 Mahatma Gandhi Road, Bengaluru 560001
```

Even if the complete strings are different:

```text
PIN = match
House number = match
City ≈ match
Road tokens = match
```

Together these provide strong evidence.

---

# 14. Country Features

Use country information carefully.

Useful examples:

```text
US:
ZIP format
state abbreviations

India:
PIN format
Indian address patterns
landmark tokens

Generic:
country-agnostic normalization
```

However, France is unseen in training.

Therefore the core model should rely primarily on **country-agnostic features** such as:

- name similarity
- token similarity
- address component overlap
- number matching
- character similarity

Country-specific rules should be supplementary.

---

# 15. Proposed Feature Set

For each candidate pair:

```text
NAME
├── normalized exact match
├── Jaccard
├── Levenshtein similarity
├── TF-IDF cosine similarity
├── common substring/LCS
└── phonetic match

ADDRESS
├── normalized exact match
├── token overlap
├── Jaccard
├── city similarity
├── state similarity
├── PIN/ZIP match
├── house-number match
└── address similarity

COUNTRY
├── country match
└── country-specific validation features

META
├── source pair (S1-S2 / S1-S3)
└── optional missingness indicators
```

We should not blindly keep every feature. Feature importance and validation should determine which features actually help.

---

# 16. Model

## Primary Model

Start with:

- **XGBoost**
- **LightGBM**

Input:

```text
Candidate pair
      |
      v
Similarity features
      |
      v
XGBoost / LightGBM
      |
      v
P(match)
```

The model outputs a probability or score representing how likely the pair is to be a true match.

### Why Gradient-Boosted Trees?

They are suitable because:

- features are mostly numerical similarity scores
- nonlinear relationships can be learned
- missing values can be handled
- feature importance is available
- training/inference are relatively efficient
- threshold tuning is straightforward

Other candidates:

- CatBoost
- Siamese networks
- Transformers

These should be experiments rather than the initial solution.

---

# 17. F0.5 and Threshold Selection

The challenge uses **F0.5**, which weights precision more heavily than recall.

```text
F0.5 = 1.25 × Precision × Recall
       --------------------------------
       0.25 × Precision + Recall
```

Interpretation:

- **Precision:** Of the predicted matches, how many are correct?
- **Recall:** Of the true matches, how many did we find?

Because F0.5 emphasizes precision, false matches are particularly costly.

Therefore we should not automatically use:

```text
probability > 0.50
```

Instead:

```text
0.10
0.20
0.30
...
0.90
```

can be tested on validation data.

Choose the threshold that performs best on the competition metric while preserving sensible behavior for singleton entities.

---

# 18. Singleton Handling

A singleton is a Source 1 entity with:

```text
zero matching records
```

The system must be able to say:

```text
"No matching Source 2/3 record."
```

This is important because always selecting the highest-scoring candidate would create false matches.

A safer decision process is:

```text
Best candidate score
        |
        +-- below threshold --> NO MATCH
        |
        +-- above threshold --> MATCH
```

Additional margin rules can be tested:

```text
Best score = 0.91
Second best = 0.42
        -> strong candidate

Best score = 0.76
Second best = 0.75
        -> ambiguous
```

The second case may require a more conservative decision.

---

# 19. Recommended End-to-End Architecture

```text
                    TRAINING DATA
                         |
                         v
              Data Loading / Profiling
                         |
                         v
                 Text Normalization
                         |
                         v
                 Candidate Blocking
              /        |        |       \
         Country    Address   Phonetic   TF-IDF
              \        |        |       /
               \       |        |      /
                    Candidate UNION
                         |
                         v
                Candidate Pair Table
                         |
                         v
                 Feature Engineering
                         |
                         v
                 XGBoost / LightGBM
                         |
                         v
                 Match Probability
                         |
                         v
                Threshold Optimization
                         |
              +----------+----------+
              |                     |
           MATCH                 NO MATCH
              |                     |
              +----------+----------+
                         |
                         v
                  Final Submission
```

---

# 20. Optional Second-Level Verification

After the baseline works, ambiguous pairs can be sent to a stronger verifier.

Possible verifier:

- Transformer cross-encoder
- Siamese/bi-encoder model

Example:

```text
Blocking
   ↓
XGBoost
   ↓
High-confidence → accept
Low-confidence → reject
Ambiguous → Transformer verifier
```

This is much more computationally practical than running a Transformer on every possible pair.

---

# 21. Graph-Based Approach — Optional, Not First Priority

A graph can represent:

```text
Node = business record
Edge = strong similarity between two records
Weight = similarity/confidence
```

Then clustering can identify groups of records that may represent the same entity.

Possible methods:

- connected components
- Louvain
- Leiden
- hierarchical clustering

Example:

```text
S1-A ---- S2-X
 |          |
 |          |
 S3-Y ---- S3-Z
```

If all connections are sufficiently strong, they may form one business cluster.

### Advantages

- captures transitive relationships
- can handle multiple records for one entity
- provides cluster-level consistency

### Problems

For millions of records:

- graph construction is expensive
- memory usage can become large
- threshold selection becomes difficult
- erroneous edges can propagate through a cluster

Therefore:

> **Do not make graph clustering the initial core solution.**

Use it later for **high-confidence consistency checking** if experiments show measurable improvement.

---

# 22. Transformer-Based Approach — Optional

A Transformer can learn semantic similarity between two records.

Example input:

```text
[Source 1]
ABC Technologies Pvt Ltd
12 MG Road Bangalore

[Candidate]
ABC Technology Private Limited
12 Mahatma Gandhi Road Bengaluru
```

A cross-encoder can output a match score.

Advantages:

- captures semantic similarity
- can learn complex variations
- potentially useful for ambiguous records

Problems:

- expensive for millions of candidate pairs
- slower inference
- higher memory/compute requirements
- potential overfitting to US/India patterns

Therefore:

> **Use Transformers as a verifier for difficult cases, not as the first model.**

---

# 23. Feasibility Assessment

## Overall

The proposed **blocking + feature engineering + gradient-boosted classifier** approach is feasible.

The main feasibility condition is:

> **Never perform all-pairs comparison.**

### Feasible

```text
Millions of records
      ↓
Efficient blocking
      ↓
Manageable candidate pairs
      ↓
Feature computation
      ↓
XGBoost/LightGBM
```

### Not feasible

```text
2M records × millions of candidates
      ↓
Full pairwise similarity
```

That would be computationally excessive.

---

# 24. Main Computational Risks

## 1. Blocking

Poor blocking can eliminate true matches.

This is the highest-risk part because a removed candidate can never be recovered.

## 2. TF-IDF

Do not build massive all-pairs similarity matrices.

Use TF-IDF only for candidate retrieval/blocking, preferably with sparse representations and top-k retrieval.

## 3. String Similarity

Calculating expensive edit-distance features for millions of unnecessary pairs will be slow.

Only calculate them after blocking.

## 4. Memory

Files are hundreds of MB each, and the combined dataset is much larger.

Use:

- chunked reading
- selected columns
- efficient dtypes
- sparse matrices
- cached intermediate files

## 5. Transformer Inference

Avoid applying a Transformer to every possible pair.

Restrict it to ambiguous candidates.

---

# 25. Efficiency Strategy

Use:

1. **Chunked file loading**
2. **Vectorized string operations where possible**
3. **Multiple blocking indexes**
4. **Sparse TF-IDF**
5. **Top-k candidate retrieval**
6. **Parallel feature computation where useful**
7. **Caching**
8. **Intermediate candidate files**
9. **Memory profiling**
10. **Start on a subset before scaling**

Recommended development sequence:

```text
Small subset
    ↓
Validate logic
    ↓
Measure candidate counts
    ↓
Measure recall of blocking
    ↓
Scale to full dataset
```

---

# 26. Validation Strategy

The validation set should contain:

- US examples
- India examples
- singleton examples

If temporal information is meaningful, a time-based split can also be evaluated.

Monitor:

| Metric | Purpose |
|---|---|
| F0.5 | Primary competition metric |
| Precision | Avoid false matches |
| Recall | Avoid missed matches |
| F1 | General reference |
| Singleton accuracy | Evaluate no-match decisions |
| Per-country metrics | Detect country-specific failures |

Also inspect:

- false positives
- false negatives
- ambiguous pairs
- blocking misses
- performance by source pair

---

# 27. Critical Validation: Blocking Recall

Before spending significant effort on the classifier, measure:

> **What percentage of known true matches survive blocking?**

Example:

```text
100,000 true matches
        |
        | Blocking
        v
97,000 retained
```

Blocking recall:

```text
97%
```

The classifier cannot exceed the effective ceiling created by candidate generation.

Therefore blocking recall should be treated as a first-class metric.

---

# 28. Zero-Shot France Strategy

France does not appear in training.

We should therefore avoid assumptions such as:

```text
"India always has PIN"
"US always has ZIP"
```

Instead prioritize universal features:

- normalized names
- character similarity
- token similarity
- phonetic similarity
- address component overlap
- numeric component matching
- generic country-independent rules

Country-specific features can remain as additional signals.

We should also evaluate whether the model behaves sensibly when country-specific features are unavailable or unfamiliar.

---

# 29. Risk Mitigation

| Risk | Mitigation |
|---|---|
| True match removed by blocking | Multiple blocking passes + union |
| Too many candidates | Tighten secondary retrieval/top-k limits |
| False matches | Conservative threshold |
| Singleton incorrectly matched | Explicit no-match logic |
| France performance poor | Country-agnostic features |
| Full dataset too slow | Chunking + sparse retrieval + caching |
| Transformer too expensive | Use only for ambiguous cases |
| Graph too expensive | Use only after baseline |
| Overfitting | Country-aware validation + error analysis |

---

# 30. Development Roadmap

## Week 1 — Foundation

- [ ] Load and profile datasets
- [ ] Inspect class/match distribution
- [ ] Normalize names and addresses
- [ ] Implement simple blocking
- [ ] Build rule-based baseline
- [ ] Build validation pipeline
- [ ] Measure initial F0.5
- [ ] Measure blocking recall

## Week 2 — Candidate Generation + Features

- [ ] Multi-pass blocking
- [ ] Phonetic matching
- [ ] Address component extraction
- [ ] Jaccard similarity
- [ ] Levenshtein similarity
- [ ] TF-IDF retrieval
- [ ] Address similarity features
- [ ] Feature importance analysis

## Week 3 — ML Model

- [ ] Train XGBoost/LightGBM
- [ ] Tune hyperparameters
- [ ] Optimize F0.5
- [ ] Tune match threshold
- [ ] Analyze false positives/negatives
- [ ] Evaluate singleton behavior
- [ ] Evaluate per-country performance

## Week 4 — Refinement

- [ ] Test France/zero-shot behavior
- [ ] Improve blocking
- [ ] Add ambiguity handling
- [ ] Optional Transformer verification
- [ ] Optional graph consistency checks
- [ ] Train final model
- [ ] Validate submission
- [ ] Prepare documentation

---

# 31. Model Development Priority

Recommended effort order:

### Priority 1 — Blocking

This determines the candidate pool and recall ceiling.

### Priority 2 — Normalization + Features

Good features make noisy records comparable.

### Priority 3 — XGBoost/LightGBM

Strong baseline for pairwise classification.

### Priority 4 — Threshold Optimization

Optimize specifically for F0.5.

### Priority 5 — Error Analysis

Study why matches fail.

### Priority 6 — Advanced Models

Only after the baseline has been measured:

- Transformer verifier
- Siamese model
- Graph consistency
- Ensemble

---

# 32. Architecture Comparison

| Approach | Role | Feasibility | Recommendation |
|---|---|---|---|
| Blocking + XGBoost/LightGBM | Main pipeline | High | **Start here** |
| Transformer cross-encoder | Ambiguous verification | Medium | Add later |
| Siamese/bi-encoder | Semantic retrieval | Medium | Experiment later |
| Graph clustering | Consistency/cluster analysis | Low–Medium at full scale | Optional |
| Full end-to-end Transformer | Entire problem | Low for this scale | Not initial approach |

---

# 33. Minimum Viable Solution

A practical MVP is:

```text
Normalization
     ↓
Country + Address + Phonetic blocking
     ↓
String + Address features
     ↓
XGBoost
     ↓
Threshold tuning
     ↓
Submission
```

Do not wait for Transformers or graph algorithms before building this baseline.

The baseline gives us:

- measurable F0.5
- blocking recall
- runtime
- candidate counts
- error examples

Those measurements tell us what actually needs improvement.

---

# 34. Target Architecture

After the baseline:

```text
                    RAW TSV FILES
                         |
                         v
              Chunked Data Loading
                         |
                         v
              Normalization / Parsing
                         |
                         v
                 Multi-Pass Blocking
          /          |          |          \
      Country     Address    Phonetic     TF-IDF
          \          |          |          /
           \         |          |         /
                Candidate UNION
                         |
                         v
                Pairwise Features
                         |
                         v
                XGBoost / LightGBM
                         |
             +-----------+-----------+
             |                       |
       High Confidence          Ambiguous
             |                       |
             |                Transformer verifier
             |                       |
             +-----------+-----------+
                         |
                         v
              Conservative Decisions
                         |
                         v
              Singleton / Consistency
                         |
                         v
                  Submission TSV
```

Graph-based consistency checking can be inserted after classification if experiments justify it.

---

# 35. Competition Constraints

The plan assumes the following restrictions from the challenge documentation:

- No external data lookup
- No external APIs
- No commercial entity-resolution services
- No government database lookup
- No external data augmentation
- Must handle the unseen France test country
- Must use the provided training data
- All test entities must appear in the submission

This means the solution must derive its matching capability from the supplied records and learned/general string and address patterns.

---

# 36. Submission Requirements

Expected output/package described in the current plan:

```text
output/
├── matching_results.tsv
└── candidate_pairs.tsv

code/
└── business_entity_resolution/
    ├── src/
    ├── README.md
    └── requirements.txt
```

Also include:

- methodology document
- exact required column names
- tab-separated output
- correct handling of singleton/no-match cases
- all test entities in the submission

Always validate the final submission using the provided validation script where applicable.

---

# 37. Key Terms — Quick Reference

| Term | Simple meaning |
|---|---|
| Entity Resolution | Finding records that represent the same real-world entity |
| Blocking | Removing obviously impossible comparisons before expensive matching |
| Candidate Pair | Two records selected for detailed comparison |
| Multi-pass Blocking | Generate candidates using several independent blocking rules |
| Blocking Recall | % of true matches that survive blocking |
| Feature Engineering | Converting record differences/similarities into model inputs |
| Levenshtein | Character edit distance |
| Jaccard | Token/set overlap |
| TF-IDF | Text representation emphasizing informative words |
| Cosine Similarity | Similarity between vector representations |
| Phonetic Encoding | Representation based on approximate pronunciation |
| XGBoost | Gradient-boosted decision-tree model |
| Threshold | Score above which we call a pair a match |
| Singleton | Entity with no matching Source 2/3 records |
| False Positive | Saying two businesses match when they do not |
| False Negative | Missing a real business match |
| Precision | Correct predicted matches / all predicted matches |
| Recall | Correct predicted matches / all actual matches |
| F0.5 | Metric that places greater emphasis on precision |
| Zero-Shot | Handling a category/country not represented during training |
| Cross-Encoder | Transformer that scores a pair together |
| Graph Clustering | Grouping records based on similarity connections |

---

# 38. Final Recommendation

The plan is **feasible**, but it should be simplified into a strong baseline-first strategy.

### Core system

```text
Normalization
      ↓
Multi-pass blocking
      ↓
Candidate pairs
      ↓
Similarity/address features
      ↓
XGBoost / LightGBM
      ↓
F0.5 threshold optimization
      ↓
Final matches + singleton handling
```

### Later improvements

```text
Ambiguous pairs
      ↓
Transformer verification
```

and potentially:

```text
High-confidence matches
      ↓
Graph consistency checking
```

The critical engineering principle is:

> **Do not spend compute on pairs that blocking can eliminate, and do not let blocking eliminate true matches.**

The first experiment should therefore measure **blocking recall, candidate volume, runtime, and baseline F0.5** before adding advanced models.

---

# 39. Important Caveats

Some implementation details in this plan are strategic proposals rather than confirmed competition requirements. In particular:

- exact candidate counts should be determined experimentally
- the exact blocking keys should be validated on the supplied training data
- the usefulness of country-specific features should be measured
- the value of Transformers/graphs should be demonstrated by validation results
- target F0.5 values such as `0.60`, `0.75`, and `0.85` should be treated as internal development goals, not guaranteed competition outcomes
- the exact submission schema should always be checked against the official challenge documentation

