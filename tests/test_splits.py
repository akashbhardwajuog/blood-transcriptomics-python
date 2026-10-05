import pandas as pd
from bloodml.common import PROCESSED, require

def test_no_patient_leakage():
    """Proves mathematically that no patient exists in both training and testing cohorts."""
    meta_train = pd.read_csv(require(PROCESSED / "clinical_train.csv"), index_col=0)
    meta_test = pd.read_csv(require(PROCESSED / "clinical_test.csv"), index_col=0)
    
    train_patients = set(meta_train['subject_id'].unique())
    test_patients = set(meta_test['subject_id'].unique())
    
    intersection = train_patients.intersection(test_patients)
    assert len(intersection) == 0, f"DATA LEAKAGE DETECTED! Overlapping patients: {intersection}"