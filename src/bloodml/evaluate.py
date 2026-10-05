import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import roc_curve, auc
from .common import PROCESSED, MODELS, FIGURES, TABLES, require

def run_evaluation():
    print("Loading data and models...")
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    
    # Load test data
    meta_test = pd.read_csv(require(PROCESSED / "clinical_test.csv"), index_col=0)
    expr_test = pd.read_csv(require(PROCESSED / "expression_test.csv"), index_col=0)
    y_test = meta_test['label'].values
    
    # Load Scikit-Learn Pipeline
    pipe = joblib.load(require(MODELS / "sklearn_pca_logreg.joblib"))
    scaler = pipe.named_steps['scaler']
    pca = pipe.named_steps['pca']
    clf = pipe.named_steps['clf']
    
    # 1. Plot PCA Scatter
    expr_scaled = scaler.transform(expr_test)
    expr_pca = pca.transform(expr_scaled)
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=expr_pca[:, 0], y=expr_pca[:, 1], hue=y_test, palette=['blue', 'red'], s=100)
    plt.title("PCA of Blood Transcriptomes (PC1 vs PC2)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Diagnosis (0=Viral, 1=Bacterial)")
    plt.tight_layout()
    plt.savefig(FIGURES / "pca_scatter.png", dpi=300)
    plt.close()
    
    # 2. Extract Top Contributing Genes 
    gene_weights = clf.coef_[0] @ pca.components_
    gene_names = expr_test.columns
    
    importance_df = pd.DataFrame({'Gene': gene_names, 'Weight': gene_weights})
    importance_df['Absolute_Weight'] = np.abs(importance_df['Weight'])
    top_genes = importance_df.sort_values(by='Absolute_Weight', ascending=False).head(20)
    top_genes.to_csv(TABLES / "top_20_biomarker_genes.csv", index=False)
    
    # Plot Top Genes
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Weight', y='Gene', data=top_genes, palette='vlag')
    plt.title("Top 20 Genes Differentiating Bacterial vs. Viral Infection")
    plt.xlabel("Log-Odds Weight")
    plt.tight_layout()
    plt.savefig(FIGURES / "top_genes_weights.png", dpi=300)
    plt.close()
    
    print("Evaluation complete! Figures and tables saved to reports/.")

if __name__ == "__main__":
    run_evaluation()