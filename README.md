# High-Dimensional Blood Transcriptomics Pipeline (GSE63990)



An end-to-end, memory-efficient Python pipeline designed to classify bacterial vs. viral respiratory infections using raw GEO microarray matrices (Affymetrix GPL571).



## Project Overview

This repository demonstrates production-grade computational genomics in Python. It parses high-dimensional unstructured GEO data (22,277 genes) and enforces strict defensive programming to prevent data leakage—a critical failure point in clinical bioinformatics.



**Key Engineering Implementations:**

* **Memory-Efficient Ingestion:** Custom parser reads compressed `.txt.gz` GEO matrices line-by-line, dynamically isolating unstructured clinical metadata from the expression matrix without crashing system RAM.

* **Leakage Prevention:** Aggressively filters repeated patient visits (taking only the earliest temporal array per subject) prior to stratified 80/20 splitting, ensuring zero patient overlap between training and testing cohorts.

* **Dimensionality Reduction:** Utilizes Scikit-Learn `Pipeline` architectures to dynamically apply Principal Component Analysis (PCA) scaled strictly to the inner cross-validation folds, preventing mathematical impossibility errors on small-N clinical cohorts.

* **Deep Learning Regularization:** Benchmarks a PyTorch Feed-Forward Neural Network utilizing `BatchNorm1d` and heavy `Dropout` (0.5) against an L2-Regularized Logistic Regression to counter the extreme curse of dimensionality inherent to transcriptomics.



## Reproducibility Guide

1. **Install environment:** `python -m pip install -e .`

2. **Ingest data:** `python scripts/download_data.py` 

3. **Run Quality Control:** `python -m bloodml.prepare`

4. **Lock Splits:** `python -m bloodml.splits`

5. **Train Models:** `python -m bloodml.train_sklearn` and `python -m bloodml.train_torch`

6. **Evaluate Biological Biomarkers:** `python -m bloodml.evaluate`

