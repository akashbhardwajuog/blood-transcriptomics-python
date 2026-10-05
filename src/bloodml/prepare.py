import pandas as pd
import numpy as np
import gzip
from .common import RAW, INTERIM, load_config, require

def parse_geo_matrix(filepath):
    """Memory-efficient parser for GEO Series Matrix files."""
    metadata_dict = {}
    expression_lines = []
    
    with gzip.open(filepath, 'rt', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith('!Sample_'):
                parts = line.strip().split('\t')
                key = parts[0].replace('!Sample_', '')
                metadata_dict[key] = [p.strip('"') for p in parts[1:]]
            elif not line.startswith('!') and not line.startswith('^') and line.strip():
                expression_lines.append(line.strip().split('\t'))
                
    # 1. Build Clinical Metadata
    meta_df = pd.DataFrame(metadata_dict)
    if 'geo_accession' in meta_df.columns:
        meta_df.set_index('geo_accession', inplace=True)
        
    # 2. Build Gene Expression Matrix (Transposed to Samples x Genes)
    exp_df = pd.DataFrame(expression_lines[1:], columns=[p.strip('"') for p in expression_lines[0]])
    exp_df.set_index(exp_df.columns[0], inplace=True)
    exp_df = exp_df.apply(pd.to_numeric, errors='coerce')
    exp_df = exp_df.T 
    
    # Align rows
    exp_df.index = meta_df.index
    return meta_df, exp_df

def run_prepare():
    cfg = load_config()
    print("Parsing raw GEO matrix. This may take a moment...")
    meta_df, exp_df = parse_geo_matrix(require(RAW / "GSE63990_series_matrix.txt.gz"))
    
    print(f"Raw array shape: {exp_df.shape[0]} arrays, {exp_df.shape[1]} genes")
    
    # Extract diagnosis (Viral = 0, Bacterial = 1) from GEO metadata
    # We search the characteristics columns for the diagnosis strings
    char_cols = [c for c in meta_df.columns if 'characteristics' in c]
    meta_string = meta_df[char_cols].astype(str).apply(lambda x: ' '.join(x).lower(), axis=1)
    
    meta_df['label'] = np.nan
    meta_df.loc[meta_string.str.contains('bacterial'), 'label'] = 1
    meta_df.loc[meta_string.str.contains('viral'), 'label'] = 0
    
    # Extract Patient IDs to prevent leakage (GSE63990 usually lists subject IDs)
    # Using defensive proxy mapping if exact formatting varies
    meta_df['subject_id'] = meta_df['title'].str.extract(r'(Subject_.*?_.*?|Pt_\d+)')[0]
    meta_df['subject_id'] = meta_df['subject_id'].fillna(meta_df.index.to_series()) # Fallback to array ID
    
    # Filter out Ambiguous/Healthy labels
    valid_mask = meta_df['label'].notna()
    meta_df = meta_df[valid_mask]
    exp_df = exp_df[valid_mask]
    
    # RULE: Keep only the earliest array per subject to prevent data leakage
    meta_df = meta_df.drop_duplicates(subset=['subject_id'], keep='first')
    exp_df = exp_df.loc[meta_df.index]
    
    print(f"Final QC shape: {exp_df.shape[0]} unique patients, {exp_df.shape[1]} genes.")
    
    # Save processed matrices
    meta_df.to_csv(INTERIM / "clinical_qc.csv")
    exp_df.to_csv(INTERIM / "expression_qc.csv")
    print("Quality Control complete. Cleaned matrices saved to interim.")

if __name__ == "__main__":
    run_prepare()