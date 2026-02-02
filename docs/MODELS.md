# Model Documentation

This document describes the machine learning models available in the PCORI system.

## Supported Models

| Model | Type | Library | Best For |
|-------|------|---------|----------|
| LightGBM | Gradient Boosting | lightgbm | Tabular data, fast iteration |
| Logistic Regression | Linear | scikit-learn | Baseline, interpretability |
| Random Forest | Ensemble | scikit-learn | Feature importance |
| Decision Tree | Tree | scikit-learn | Explainable rules |
| LSTM | Deep Learning | PyTorch | Temporal sequences |
| GRU | Deep Learning | PyTorch | Temporal sequences |
| BiLSTM | Deep Learning | PyTorch | Bidirectional patterns |

## Model Configurations

### LightGBM (Default)

```python
params = {
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}
```

### LSTM/GRU

```python
config = {
    'hidden_size': 128,
    'num_layers': 2,
    'dropout': 0.3,
    'bidirectional': False,  # True for BiLSTM
    'learning_rate': 1e-3,
    'batch_size': 64,
    'epochs': 50
}
```

## Training

### Command Line

```bash
cd pipeline-pcori

# LightGBM
python train.py --dataset ./data/synth --model lightgbm

# LSTM
python train.py --dataset ./data/synth --model lstm \
    --epochs 10 --hidden_size 128 --batch_size 128

# Random Forest
python train.py --dataset ./data/synth --model rf
```

### Via Dashboard

1. Open SITL Dashboard
2. Navigate to Training UI
3. Select model type and parameters
4. Click "Train"

## Evaluation Metrics

All models report:
- **AUROC**: Area Under ROC Curve
- **AUPRC**: Area Under Precision-Recall Curve
- **Accuracy**: Overall classification accuracy
- **F1 Score**: Harmonic mean of precision and recall

## Model Artifacts

Trained models are saved to `runs/<timestamp>/`:

```
runs/
└── 2026-02-01_12-00-00/
    ├── model.pkl          # Trained model
    ├── config.json        # Training configuration
    ├── metrics.json       # Evaluation metrics
    └── feature_importance.csv
```

## Explainability

All models support SHAP explanations:

```python
import shap

# For tree-based models
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# For deep learning
explainer = shap.DeepExplainer(model, X_train[:100])
shap_values = explainer.shap_values(X_test)
```

## Model Selection Guidelines

| Scenario | Recommended Model |
|----------|-------------------|
| Quick prototyping | LightGBM |
| Maximum interpretability | Logistic Regression |
| Temporal patient data | LSTM or GRU |
| Feature importance analysis | Random Forest |
| Production deployment | LightGBM or ensemble |
