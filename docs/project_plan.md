# Project Plan
- **Dataset:** GSE63990 (Affymetrix GPL571) - Blood Transcriptomics
- **Question:** Classify bacterial vs. viral respiratory infections.
- **Encoding:** 0 = viral, 1 = bacterial (positive class).
- **Primary Metric:** ROC-AUC.
- **Models:** L2-Regularized Logistic Regression (with PCA dimensionality reduction), PyTorch Deep Neural Network.
- **Predeclared Decisions:**
  - Repeated arrays: Keep one array per subject (earliest by GEO accession) to prevent data leakage.
  - Ambiguous labels: Excluded.
  - Split: 80/20 stratified by subject, seed 42, locked before modeling.