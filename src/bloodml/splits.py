import pandas as pd
from sklearn.model_selection import train_test_split
from .common import INTERIM, PROCESSED, load_config, require

def make_splits():
    cfg = load_config()
    
    print("Loading QC'd matrices...")
    meta = pd.read_csv(require(INTERIM / "clinical_qc.csv"), index_col=0)
    expr = pd.read_csv(require(INTERIM / "expression_qc.csv"), index_col=0)
    
    # Defensive check: ensure clinical rows map exactly to expression rows
    if not all(meta.index == expr.index):
        raise ValueError("Critical Error: Clinical and Expression array IDs do not match!")
        
    print(f"Total cohort size: {len(meta)} patients.")
    
    # 80/20 Stratified Split
    meta_train, meta_test, expr_train, expr_test = train_test_split(
        meta, expr,
        test_size=cfg["test_fraction"],
        stratify=meta['label'],
        random_state=cfg["seed"]
    )
    
    PROCESSED.mkdir(parents=True, exist_ok=True)
    
    # Lock the splits to disk
    meta_train.to_csv(PROCESSED / "clinical_train.csv")
    meta_test.to_csv(PROCESSED / "clinical_test.csv")
    expr_train.to_csv(PROCESSED / "expression_train.csv")
    expr_test.to_csv(PROCESSED / "expression_test.csv")
    
    print("Data successfully locked to prevent leakage.")
    print(f"Training cohort: {len(meta_train)} patients")
    print(f"Testing cohort: {len(meta_test)} patients")

if __name__ == "__main__":
    make_splits()