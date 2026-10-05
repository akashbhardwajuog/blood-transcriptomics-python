from pathlib import Path

directories = [
    "configs", "data/raw", "data/interim", "data/processed", "data/reference",
    "notebooks", "src/bloodml", "scripts", "tests", "models",
    "reports/figures", "reports/tables", "reports/logs", "docs",
]
files = [
    ".gitignore", "README.md", "requirements.txt", "pyproject.toml",
    "configs/default.json", "configs/label_map.json", "data/README.md",
    "docs/project_plan.md", "docs/data_dictionary.md", "docs/preprocessing_audit.md",
    "docs/experiment_log.md", "docs/model_card.md", "docs/learning_log.md",
    "docs/frozen_analysis.md", "reports/final_report.md",
    "src/bloodml/__init__.py", "src/bloodml/__main__.py", "src/bloodml/common.py",
    "src/bloodml/io.py", "src/bloodml/prepare.py", "src/bloodml/qc.py",
    "src/bloodml/splits.py", "src/bloodml/train_sklearn.py",
    "src/bloodml/train_torch.py", "src/bloodml/numerics.py",
    "src/bloodml/evaluate.py", "src/bloodml/annotation.py",
    "scripts/download_data.py",
    "tests/test_io.py", "tests/test_splits.py", "tests/test_numerics.py",
    "tests/test_torch.py",
]

for d in directories:
    Path(d).mkdir(parents=True, exist_ok=True)
for f in files:
    Path(f).touch(exist_ok=True)
    
print("Blood Transcriptomics project structure created.")