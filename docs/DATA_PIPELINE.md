# Data Pipeline

This document describes the data processing pipeline for the PCORI system.

## Overview

```
Raw EHR Data → Preprocessing → Feature Engineering → Model Training → Predictions
                    ↓
              Feature Store (Parquet)
                    ↓
              Patient Database (SQLite)
```

## Data Sources

### Supported Formats

| Source | Format | Use Case |
|--------|--------|----------|
| Health Facts | CSV/Parquet | Production EHR data |
| Synthetic | JSON/CSV | Development and testing |
| Custom | CSV | User-provided data |

### Data Schema

Required columns:
- `patient_id`: Unique patient identifier
- `encounter_id`: Visit/encounter identifier
- `timestamp`: Event timestamp
- `feature_*`: Feature columns
- `label`: Binary outcome (0/1)

## Pipeline Components

### 1. Data Ingestion

```python
from pipeline.data_loader import load_dataset

# Load from CSV
data = load_dataset('./data/synth', format='csv')

# Load from Parquet
data = load_dataset('./data/health_facts', format='parquet')
```

### 2. Preprocessing

- Missing value imputation
- Outlier detection
- Date/time parsing
- Categorical encoding

```python
from pipeline.preprocessing import preprocess

cleaned_data = preprocess(
    data,
    impute_strategy='median',
    encode_categoricals=True
)
```

### 3. Feature Engineering

- Temporal aggregations
- Diagnosis code grouping
- Lab value normalization
- Interaction features

```python
from pipeline.features import engineer_features

features = engineer_features(
    cleaned_data,
    spec='./configs/FeatureSpec.json'
)
```

### 4. Train/Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    features, labels,
    test_size=0.2,
    stratify=labels,
    random_state=42
)
```

## Feature Specification

Features are defined in `FeatureSpec.json`:

```json
{
  "features": [
    {
      "name": "age",
      "type": "numeric",
      "source": "demographics"
    },
    {
      "name": "diagnosis_count",
      "type": "numeric",
      "aggregation": "count",
      "source": "diagnoses"
    }
  ],
  "label": {
    "name": "overdose_30d",
    "type": "binary"
  }
}
```

## Sequential Data

For LSTM/GRU models, data is structured as sequences:

```python
# Shape: (num_patients, time_steps, num_features)
X_seq = prepare_sequences(
    data,
    T=10,  # sequence length
    features=['age', 'rx_count', 'visit_count']
)
```

## Data Quality

### Validation Checks

- Missing value rates
- Label balance
- Feature distributions
- Temporal consistency

```python
from pipeline.validation import validate_data

report = validate_data(data)
print(report.summary())
```

## Privacy Considerations

- All patient data is de-identified
- No direct identifiers (SSN, name, address)
- Date shifting applied
- Minimum cell sizes enforced

See [Research Compliance](../README.md#research-compliance--governance) for details.
