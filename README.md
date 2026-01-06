# AI-Driven Financial Fraud Detection in Digital Payments

**Course:** AI Course – Final Project (Week 17 Documentation & Submission)  
**Team:** Vedanth (1147284), Paul (1147256), Somnath (1137253)

## 1) Project Overview
This project builds a machine learning pipeline to detect **fraudulent digital payment transactions**.  
Because fraud datasets are **highly imbalanced**, we focus on metrics beyond accuracy (especially **recall/F1 for the fraud class**) and apply:
- **SMOTE** for class balancing  
- **Feature engineering** (e.g., `balanceDiffPercent`, `inNightTransaction`, `transactionCount`)  
- Baseline model: **Logistic Regression**  
- Advanced models: **XGBoost**, **LightGBM**  
- Evaluation: classification report + confusion matrix

## 2) Repository Structure (recommended)
```
.
├── notebooks/
│   └── Financial Fraud Detection Analysis.ipynb
├── data/
│   └── Fraud.csv                # not committed if too large; see dataset section below
├── outputs/
│   ├── confusion_matrix_xgb.png
│   └── confusion_matrix_lgbm.png
├── requirements.txt
├── README.md
└── ONE_PAGE_INSTRUCTIONS.pdf
```

## 3) Environment Setup
### Option A — Python (recommended)
```bash
# 1) Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt
```

### Option B — Conda
```bash
conda create -n fraud-detect python=3.10 -y
conda activate fraud-detect
pip install -r requirements.txt
```

## 4) Dataset
This project uses a `Fraud.csv` dataset (PaySim-style synthetic mobile money transactions).
Download the dataset from Kaggle (choose one that provides `Fraud.csv`), then place it here:

```
data/Fraud.csv
```

Example Kaggle sources:
- PaySim datasets on Kaggle (e.g., *Synthetic Financial Datasets For Fraud Detection* / PaySim variants)

> ⚠️ If the dataset file is large, do **NOT** commit it to GitHub. Instead, commit this folder structure and provide the dataset link here.

## 5) How to Reproduce Results
### Run the notebook
```bash
jupyter notebook
```
Open:
- `notebooks/Financial Fraud Detection Analysis.ipynb`

Run cells top-to-bottom. The notebook will:
1) Load `data/Fraud.csv`
2) Preprocess + explore data
3) Train baseline + advanced models
4) Print metrics and show confusion matrices

## 6) Train Your Own Model (same workflow)
In the notebook, you can modify:
- Train/test split ratio
- SMOTE parameters
- Model hyperparameters (XGBoost/LightGBM)
Then re-run training cells and compare metrics.

## 7) Results Summary (high level)
- Accuracy alone is misleading under heavy class imbalance.
- After SMOTE + feature engineering + advanced models, fraud detection recall/F1 improves versus baseline.
- Confusion matrices are used to visualize fraud vs non-fraud classification.

## 8) Notes / Limitations
- Synthetic dataset may not capture all real-world fraud patterns.
- Performance depends on class imbalance, thresholds, and feature quality.
- Consider threshold tuning and cost-sensitive learning for further improvements.
