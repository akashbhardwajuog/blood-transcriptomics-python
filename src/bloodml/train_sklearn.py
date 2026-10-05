import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, classification_report
from .common import PROCESSED, MODELS, load_config, require

def run_sklearn():
    cfg = load_config()
    
    print("Loading locked datasets...")
    meta_train = pd.read_csv(require(PROCESSED / "clinical_train.csv"), index_col=0)
    expr_train = pd.read_csv(require(PROCESSED / "expression_train.csv"), index_col=0)
    
    meta_test = pd.read_csv(require(PROCESSED / "clinical_test.csv"), index_col=0)
    expr_test = pd.read_csv(require(PROCESSED / "expression_test.csv"), index_col=0)
    
    y_train = meta_train['label'].values
    y_test = meta_test['label'].values
    
    # The Pipeline: Scale -> Shrink (PCA) -> Classify
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(random_state=cfg["seed"])),
        ("clf", LogisticRegression(penalty="l2", solver="lbfgs", max_iter=1000, random_state=cfg["seed"]))
    ])
    
# Drop Cross-Validation folds to 2 so it doesn't crash on small data
    cv_folds = min(cfg.get("cv_folds", 5), 2)
    
    # Dynamic guardrails for our tiny 8-patient cohort
    # During 2-fold CV, the 8 training samples are split into 4 for training, 4 for validation.
    # PCA components cannot exceed this inner training size.
    inner_train_size = len(y_train) // cv_folds
    max_pca = min(inner_train_size, max(cfg["pca_components"]))
    
    pca_grid = [x for x in cfg["pca_components"] if x <= max_pca]
    if not pca_grid: pca_grid = [max_pca]
    
    param_grid = {
        "pca__n_components": list(set(pca_grid)),
        "clf__C": cfg["logistic_c_values"]
    }
    
    print(f"Tuning PCA and Logistic Regression via {cv_folds}-fold Cross Validation...")
    grid = GridSearchCV(pipe, param_grid, cv=cv_folds, scoring="accuracy", n_jobs=-1)
    
    # Fit strictly on training data
    grid.fit(expr_train, y_train)
    
    print(f"Best hyperparameters: {grid.best_params_}")
    
    # Evaluate on untouched test set
    best_model = grid.best_estimator_
    preds = best_model.predict(expr_test)
    probs = best_model.predict_proba(expr_test)[:, 1]
    
    print("\n--- Optimized PCA + Logistic Regression ---")
    try:
        print(f"ROC-AUC: {roc_auc_score(y_test, probs):.3f}")
    except ValueError:
        print("ROC-AUC not defined (test set only has one class).")
        
    print(classification_report(y_test, preds, zero_division=0))
    
    # Save the pipeline weights
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODELS / "sklearn_pca_logreg.joblib")
    print("Transcriptomics Scikit-Learn model saved successfully.")

if __name__ == "__main__":
    run_sklearn()