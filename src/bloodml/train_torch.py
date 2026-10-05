import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, classification_report
import joblib
from .common import PROCESSED, MODELS, load_config, require

class TranscriptomicNN(nn.Module):
    """A heavily regularized Feed-Forward Neural Network for small-N genomics."""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5), # Drops 50% of neurons to prevent overfitting
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x)

def run_torch():
    cfg = load_config()
    
    print("Loading locked datasets for PyTorch...")
    meta_train = pd.read_csv(require(PROCESSED / "clinical_train.csv"), index_col=0)
    expr_train = pd.read_csv(require(PROCESSED / "expression_train.csv"), index_col=0)
    
    meta_test = pd.read_csv(require(PROCESSED / "clinical_test.csv"), index_col=0)
    expr_test = pd.read_csv(require(PROCESSED / "expression_test.csv"), index_col=0)
    
    y_train = meta_train['label'].values
    y_test = meta_test['label'].values
    
    # 1. Preprocessing (Scale + PCA)
    print("Applying StandardScaler and PCA Dimensionality Reduction...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(expr_train)
    X_test_scaled = scaler.transform(expr_test)
    
    # Guardrail PCA for small N
    n_comp = min(len(y_train), cfg.get("torch_pca_components", 20))
    pca = PCA(n_components=n_comp, random_state=cfg["seed"])
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    # 2. Convert to PyTorch Tensors
    X_train_t = torch.FloatTensor(X_train_pca)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_t = torch.FloatTensor(X_test_pca)
    
    # 3. Initialize Model
    model = TranscriptomicNN(input_dim=n_comp, hidden_dim=cfg["torch_hidden_units"])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["torch_learning_rate"])
    
    # 4. Training Loop
    epochs = cfg.get("torch_max_epochs", 200)
    print(f"Training PyTorch Neural Network for {epochs} epochs...")
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
        
    # 5. Evaluation
    model.eval()
    with torch.no_grad():
        probs = model(X_test_t).squeeze().numpy()
        
    # Handle edge case where test set is so small it returns a scalar instead of array
    if probs.ndim == 0:
        probs = [probs.item()]
        
    preds = [1 if p >= 0.5 else 0 for p in probs]
    
    print("\n--- PyTorch Neural Network Results ---")
    try:
        print(f"ROC-AUC: {roc_auc_score(y_test, probs):.3f}")
    except ValueError:
        print("ROC-AUC not defined (test set only has one class).")
        
    print(classification_report(y_test, preds, zero_division=0))
    
    # 6. Save Model and Preprocessors
    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODELS / "torch_model.pt")
    joblib.dump(scaler, MODELS / "torch_scaler.joblib")
    joblib.dump(pca, MODELS / "torch_pca.joblib")
    print("Transcriptomics PyTorch model and preprocessors saved successfully.")

if __name__ == "__main__":
    run_torch()