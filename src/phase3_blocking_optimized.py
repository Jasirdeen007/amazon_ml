#!/usr/bin/env python3
"""Phase 3: Optimized Blocking - Efficient Candidate Generation"""
import pandas as pd
import numpy as np
from pathlib import Path
import sqlite3
import time
import psutil
import os

BASE = Path('d:/amazon_ml_challenge/6ab10eb3b23ba_student_resource/student_resource/dataset')
OUT = Path('d:/amazon_ml_challenge/data/processed')
CAND = Path('d:/amazon_ml_challenge/data/candidates')
DB = CAND / 'index.db'
CAND.mkdir(exist_ok=True)

CONFIG = {'chunk_size': 100000, 'batch_write': 1000, 'sample_size': 10000}

class OptimizedBlocker:
    def __init__(self, is_sample=False):
        self.is_sample = is_sample
        self.t0 = time.time()
        if DB.exists(): 
            os.remove(DB)
        self.out_file = CAND / ('candidate_pairs_sample.tsv' if is_sample else 'candidate_pairs.tsv')
        self.stats = {'s1': 0, 's2': 0, 's3': 0, 'gen': 0, 'uniq': 0, 'per_s1': []}
        self._init_db()
    
    def _mem(self): 
        return psutil.Process(os.getpid()).memory_info().rss / (1024**2)
    
    def _init_db(self):
        c = sqlite3.connect(str(DB))
        c.execute('CREATE TABLE idx (id TEXT PRIMARY KEY, nm TEXT NOT NULL, cnt TEXT NOT NULL, px TEXT NOT NULL)')
        c.execute('CREATE INDEX ix1 ON idx(nm, cnt)')
        c.execute('CREATE INDEX ix2 ON idx(px, cnt)')
        c.commit()
        c.close()

    
    def _build_index(self):
        print("\n[1/3] Building S2+S3 index...")
        c = sqlite3.connect(str(DB))
        for src, f in [('S2', OUT/'train_source2_norm.tsv'), ('S3', OUT/'train_source3_norm.tsv')]:
            print(f"  Indexing {src}...")
            df = pd.read_csv(f, sep='\t', usecols=['entity_id', 'business_name_norm', 'country'])
            # Handle NULL/NaN values
            df['business_name_norm'] = df['business_name_norm'].fillna('')
            df = df[df['business_name_norm'].str.len() > 0]
            df['px'] = df['business_name_norm'].str[:4]
            batch = [(r['entity_id'], r['business_name_norm'], r['country'], r['px']) for _, r in df.iterrows()]
            for i in range(0, len(batch), 50000):
                c.executemany('INSERT INTO idx VALUES (?, ?, ?, ?)', batch[i:i+50000])
                c.commit()
            cnt = len(df)
            self.stats['s2' if src == 'S2' else 's3'] = cnt
            print(f"    ✓ {src}: {cnt:,} (after filtering nulls)")
        c.close()
    
    def _run_blocking(self):
        print("\n[2/3] Processing S1...")
        c = sqlite3.connect(str(DB))
        seen = set()
        s1 = pd.read_csv(OUT/'train_source1_norm.tsv', sep='\t', 
                        usecols=['entity_id', 'business_name_norm', 'country'], 
                        nrows=CONFIG['sample_size'] if self.is_sample else None)
        ch = 0
        with open(self.out_file, 'w') as f:
            f.write("source1_entity_id\tmatched_entity_id\n")
            for i in range(0, len(s1), CONFIG['chunk_size']):
                ch += 1
                batch = []
                for _, r in s1.iloc[i:i+CONFIG['chunk_size']].iterrows():
                    cand = set()
                    # Exact name match
                    for row in c.execute('SELECT id FROM idx WHERE nm=? AND cnt=?', (r['business_name_norm'], r['country'])):
                        cand.add(row[0])
                    # Phonetic prefix match (fallback)
                    px = r['business_name_norm'][:4] if len(str(r['business_name_norm'])) >= 4 else ''
                    if px:
                        for row in c.execute('SELECT id FROM idx WHERE px=? AND cnt=?', (px, r['country'])):
                            cand.add(row[0])
                    self.stats['per_s1'].append(len(cand))
                    # Deduplicate and write
                    for s23 in cand:
                        p = (r['entity_id'], s23)
                        if p not in seen:
                            batch.append(p)
                            seen.add(p)
                            self.stats['gen'] += 1
                    if len(batch) >= CONFIG['batch_write']:
                        for s1_id, s23_id in batch:
                            f.write(f"{s1_id}\t{s23_id}\n")
                        batch = []
                if batch:
                    for s1_id, s23_id in batch:
                        f.write(f"{s1_id}\t{s23_id}\n")
                self.stats['s1'] += min(CONFIG['chunk_size'], len(s1) - i)
                if ch % 5 == 0:
                    print(f"  Ch{ch}: S1={self.stats['s1']:,} Cand={self.stats['gen']:,} Mem={self._mem():.0f}MB")
        c.close()
        self.stats['uniq'] = len(seen)
    
    def _recall(self):
        print("\n[3/3] Measuring recall...")
        gt = pd.read_csv(BASE/'train'/'train_ground_truth.tsv', sep='\t')
        true = set()
        for _, r in gt.iterrows():
            if not pd.isna(r['matched_entity_ids']) and r['matched_entity_ids'] != '':
                for m in str(r['matched_entity_ids']).split(','):
                    true.add((r['source1_entity_id'], m.strip()))
        cand = pd.read_csv(self.out_file, sep='\t')
        cand_set = set(zip(cand['source1_entity_id'], cand['matched_entity_id']))
        retained = len(true & cand_set)
        recall = retained / len(true) if true else 0
        print(f"  True: {len(true):,} | Retained: {retained:,} | Recall: {recall*100:.2f}%")
        return recall
    
    def _report(self):
        elapsed = time.time() - self.t0
        per_s1 = np.array(self.stats['per_s1'])
        print("\n" + "="*80)
        print("STATISTICS")
        print("="*80)
        print(f"S1: {self.stats['s1']:,} | S2: {self.stats['s2']:,} | S3: {self.stats['s3']:,}")
        print(f"Candidates: {self.stats['gen']:,} | Unique: {self.stats['uniq']:,}")
        print(f"Per S1 - Mean: {per_s1.mean():.1f} | Median: {np.median(per_s1):.1f} | P95: {np.percentile(per_s1,95):.0f} | P99: {np.percentile(per_s1,99):.0f}")
        print(f"Memory: {self._mem():.0f}MB | Time: {elapsed:.1f}s")
        print("="*80)
    
    def run(self):
        print("\n" + "="*80)
        print(f"PHASE 3 - {'SAMPLE' if self.is_sample else 'FULL'}")
        print("="*80)
        self._build_index()
        self._run_blocking()
        recall = self._recall()
        self._report()
        return recall >= 0.95



def main():
    print("\n" + "="*80)
    print("STEP 1: SAMPLE TEST (10K S1 records)")
    print("="*80)
    blocker_sample = OptimizedBlocker(is_sample=True)
    sample_ok = blocker_sample.run()
    
    if not sample_ok:
        print("\n✗ Sample recall < 95%. Review blocking strategy before scaling.")
        return 1
    
    print("\n" + "="*80)
    print("STEP 2: FULL BLOCKING (2.2M S1 records)")
    print("="*80)
    blocker_full = OptimizedBlocker(is_sample=False)
    full_ok = blocker_full.run()
    
    if full_ok:
        print("\n✓ Phase 3 COMPLETE - candidate_pairs.tsv ready for Phase 4")
    
    return 0 if full_ok else 1

if __name__ == '__main__':
    exit(main())
