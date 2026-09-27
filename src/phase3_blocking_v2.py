#!/usr/bin/env python3
"""Phase 3: Disk-Backed Blocking - Memory Efficient"""
import pandas as pd
from pathlib import Path
import sqlite3, time, psutil, os

CONFIG = {'chunk_size': 50000}
BASE = Path('d:/amazon_ml_challenge/6ab10eb3b23ba_student_resource/student_resource/dataset')
OUT = Path('d:/amazon_ml_challenge/data/processed')
CAND = Path('d:/amazon_ml_challenge/data/candidates')
DB = CAND / 'blocking.db'
CAND.mkdir(exist_ok=True)

class Blocker:
    def __init__(self):
        if DB.exists(): os.remove(DB)
        self.t0 = time.time()
        self.stats = {'s1': 0, 's2': 0, 's3': 0, 'cand': 0, 'uniq': 0}
        self._init_db()
    
    def _init_db(self):
        c = sqlite3.connect(str(DB))
        c.execute('CREATE TABLE s2_idx (id TEXT PRIMARY KEY, nm TEXT, ad TEXT, cnt TEXT, px TEXT)')
        c.execute('CREATE TABLE s3_idx (id TEXT PRIMARY KEY, nm TEXT, ad TEXT, cnt TEXT, px TEXT)')
        c.execute('CREATE TABLE cand (s1 TEXT, s2s3 TEXT, PRIMARY KEY (s1, s2s3))')
        c.execute('CREATE INDEX ix_s2 ON s2_idx(px)')
        c.execute('CREATE INDEX ix_s3 ON s3_idx(px)')
        c.commit()
        c.close()
    
    def _mem(self):
        return psutil.Process(os.getpid()).memory_info().rss / 1024**2
    
    def idx(self):
        c = sqlite3.connect(str(DB))
        for src, file, tab in [('S2', OUT/'train_source2_norm.tsv', 's2_idx'), ('S3', OUT/'train_source3_norm.tsv', 's3_idx')]:
            df = pd.read_csv(file, sep='\t', chunksize=100000)
            for ch in df:
                for _, r in ch.iterrows():
                    px = r['business_name_norm'][:4] if len(str(r['business_name_norm'])) >= 4 else ''
                    c.execute(f'INSERT INTO {tab} VALUES (?, ?, ?, ?, ?)', (r['entity_id'], r['business_name_norm'], r['business_address_norm'], r['country'], px))
                c.commit()
                self.stats['s2' if src == 'S2' else 's3'] += len(ch)
        c.close()
        print(f"✓ S2: {self.stats['s2']:,} | S3: {self.stats['s3']:,}")
    
    def block(self, s1_batch, c):
        cand = set()
        for _, r in s1_batch.iterrows():
            s1_id, nm, cnt = r['entity_id'], r['business_name_norm'], r['country']
            px = nm[:4] if len(nm) >= 4 else ''
            for row in c.execute('SELECT id FROM s2_idx WHERE nm=? AND cnt=?', (nm, cnt)):
                cand.add((s1_id, row[0]))
            for row in c.execute('SELECT id FROM s3_idx WHERE nm=? AND cnt=?', (nm, cnt)):
                cand.add((s1_id, row[0]))
            if px:
                for row in c.execute('SELECT id FROM s2_idx WHERE px=? AND cnt=?', (px, cnt)):
                    cand.add((s1_id, row[0]))
                for row in c.execute('SELECT id FROM s3_idx WHERE px=? AND cnt=?', (px, cnt)):
                    cand.add((s1_id, row[0]))
        for s1_id, s2s3_id in cand:
            try:
                c.execute('INSERT INTO cand VALUES (?, ?)', (s1_id, s2s3_id))
            except sqlite3.IntegrityError:
                pass
        c.commit()
        self.stats['cand'] += len(cand)
    
    def run(self):
        print('='*80 + '\nBLOCKING PIPELINE\n' + '='*80 + '\n')
        self.idx()
        c = sqlite3.connect(str(DB))
        print('Processing S1...')
        s1_df = pd.read_csv(OUT/'train_source1_norm.tsv', sep='\t', chunksize=CONFIG['chunk_size'])
        for ch_n, batch in enumerate(s1_df, 1):
            self.block(batch, c)
            self.stats['s1'] += len(batch)
            if ch_n % 10 == 0:
                print(f"  Ch{ch_n}: S1={self.stats['s1']:,} Cand={self.stats['cand']:,} Mem={self._mem():.0f}MB")
        c.close()
        
        print('\nExporting candidates...')
        c = sqlite3.connect(str(DB))
        df = pd.read_sql('SELECT s1, s2s3 FROM cand', c)
        c.close()
        df.columns = ['source1_entity_id', 'matched_entity_id']
        df.to_csv(CAND/'candidate_pairs.tsv', sep='\t', index=False)
        self.stats['uniq'] = len(df)
        
        print('\nMeasuring recall...')
        gt = pd.read_csv(BASE/'train'/'train_ground_truth.tsv', sep='\t')
        true = set()
        for _, r in gt.iterrows():
            if not pd.isna(r['matched_entity_ids']) and r['matched_entity_ids'] != '':
                for m in str(r['matched_entity_ids']).split(','):
                    true.add((r['source1_entity_id'], m.strip()))
        cand_set = set(zip(df['source1_entity_id'], df['matched_entity_id']))
        recall = len(true & cand_set) / len(true) if true else 0
        print(f"Recall: {recall*100:.2f}% ({len(true & cand_set):,}/{len(true):,})")
        
        elapsed = time.time() - self.t0
        print('\n' + '='*80)
        print(f"S1: {self.stats['s1']:,} | S2: {self.stats['s2']:,} | S3: {self.stats['s3']:,}")
        print(f"Candidates: {self.stats['cand']:,} | Unique: {self.stats['uniq']:,}")
        print(f"Memory: {self._mem():.0f}MB | Time: {elapsed:.0f}s")
        print('='*80)
        return 0 if recall >= 0.95 else 1

if __name__ == '__main__':
    exit(Blocker().run())
