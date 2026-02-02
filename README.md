# PCORI: Human-Centered AI for Clinical Decision Support

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PCORI Funded](https://img.shields.io/badge/PCORI-Funded-green.svg)](https://www.pcori.org/)

A comprehensive machine learning infrastructure for **Opioid Use Disorder (OUD) and Overdose (OD) risk prediction**, developed as part of a PCORI-funded research initiative at Stony Brook University. This project provides end-to-end tools for clinical risk modeling, from data preprocessing and feature selection to interactive dashboards for clinician validation.

## Project Overview

Preventable opioid overdose deaths remain a critical public health challenge. This project addresses the gap between ML model development and clinical adoption through:

- **Transparent Models**: Interpretable predictions with SHAP-based explanations
- **Clinician-in-the-Loop Validation**: Interactive dashboards for clinical review
- **Production-Ready Pipeline**: Scalable data processing for large EHR datasets
- **Multi-Model Support**: Traditional ML (LightGBM, Random Forest) and deep learning (LSTM, GRU)

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PCORI ML Infrastructure                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────────┐ │
│  │   Raw EHR    │──▶│  Pipeline    │──▶│   Feature Store          │ │
│  │   Data       │   │  (ETL)       │   │   (Parquet/SQLite)       │ │
│  └──────────────┘   └──────────────┘   └──────────────────────────┘ │
│                                                  │                   │
│                           ┌──────────────────────┘                   │
│                           ▼                                          │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Model Training Layer                       │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐              │   │
│  │  │ LightGBM   │  │   LSTM     │  │ Logistic   │  ...         │   │
│  │  │            │  │   GRU      │  │ Regression │              │   │
│  │  └────────────┘  └────────────┘  └────────────┘              │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                           │                                          │
│                           ▼                                          │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                 Explainability Layer (SHAP)                   │   │
│  │  • Global feature importance  • Per-patient explanations      │   │
│  │  • LLM-generated summaries    • Audit logging                 │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                           │                                          │
│                           ▼                                          │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              SITL Dashboard (Clinician Interface)             │   │
│  │  • Cohort Builder     • Training UI      • Patient Explorer   │   │
│  │  • Risk Scores        • AI Chat          • Export Tools       │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Components

| Component | Description | Documentation |
|-----------|-------------|---------------|
| **[SITL Dashboard](SITL_Dashboard_PCORI/)** | Interactive web platform for clinician-in-the-loop model validation | [README](SITL_Dashboard_PCORI/README.md) |
| **[Pipeline](pipeline-pcori/)** | Data preprocessing, feature engineering, and model training pipeline | [README](pipeline-pcori/README.md) |
| **[Feature Selection](feature_selection/)** | Multiple feature selection methods (NTK, LightGBM, Elastic Net, LLM) | [docs/FEATURE_SELECTION.md](docs/FEATURE_SELECTION.md) |
| **[Model](model/)** | Trained model artifacts and evaluation results | [docs/MODELS.md](docs/MODELS.md) |

## Quick Start

### Prerequisites

- Python 3.8+
- pip or conda

### Installation

```bash
# Clone the repository
git clone https://github.com/StonyBrookDB/PCORI.git
cd PCORI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the SITL Dashboard

```bash
cd SITL_Dashboard_PCORI

# Install dashboard dependencies
pip install -r requirements.txt

# Initialize the database (if using synthetic data)
python backend/init_db.py

# Start the server
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8503

# Open http://localhost:8503 in your browser
```

### Running the Training Pipeline

```bash
cd pipeline-pcori

# Train with synthetic data
python train.py \
  --dataset ./data/synth \
  --spec ./data/synth/FeatureSpec.json \
  --model lightgbm

# Available models: logreg, lightgbm, rf, dt, lstm, gru, bilstm
```

## Supported Models

| Model | Type | Use Case | Training Time |
|-------|------|----------|---------------|
| LightGBM | Gradient Boosting | Tabular data, fast iteration | <1 second |
| Logistic Regression | Linear | Baseline, interpretability | <1 second |
| Random Forest | Ensemble | Feature importance | ~seconds |
| Decision Tree | Tree | Explainable rules | <1 second |
| LSTM | Deep Learning | Temporal sequences | Minutes |
| GRU | Deep Learning | Temporal sequences | Minutes |
| BiLSTM | Deep Learning | Bidirectional patterns | Minutes |

## Feature Selection Methods

This project implements multiple feature selection approaches for identifying predictive clinical features:

| Method | Description |
|--------|-------------|
| **NTK-Inspired** | Neural Tangent Kernel sensitivity analysis |
| **LightGBM** | Gradient boosting feature importance |
| **Elastic Net** | L1/L2 regularized coefficients |
| **Recurrence Enrichment** | Condition recurrence patterns |
| **LLM-Guided** | Large language model feature ranking |

Results are combined using ensemble voting (features appearing in 3+ methods).

## Project Structure

```
PCORI/
├── SITL_Dashboard_PCORI/    # Stakeholder-in-the-Loop web dashboard
│   ├── backend/             # FastAPI server
│   ├── frontend/            # Web interface
│   ├── data/                # Local database
│   └── tests/               # API tests
├── pipeline-pcori/          # ML training pipeline
│   ├── models/              # Model implementations
│   ├── datasets/            # Data loaders
│   ├── configs/             # Training configurations
│   └── tools/               # Evaluation utilities
├── feature_selection/       # Feature selection methods
│   ├── diagnosis/           # Diagnosis-based features
│   └── labs/                # Lab-based features
├── docs/                    # Documentation
│   ├── ARCHITECTURE.md      # System architecture
│   ├── INSTALLATION.md      # Detailed setup guide
│   ├── DATA_PIPELINE.md     # Data processing details
│   └── API.md               # API reference
└── model/                   # Trained model artifacts
```

## Data Sources

This project is designed to work with:

- **Cerner Health Facts**: De-identified EHR data (requires license)
- **Synthetic Data**: Included for development and testing
- **Custom Data**: Any tabular patient data with compatible schema

## Documentation

- [Architecture Overview](docs/ARCHITECTURE.md)
- [Installation Guide](docs/INSTALLATION.md)
- [Data Pipeline](docs/DATA_PIPELINE.md)
- [API Reference](docs/API.md)
- [Contributing Guidelines](CONTRIBUTING.md)

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific component tests
pytest SITL_Dashboard_PCORI/tests/ -v
pytest pipeline-pcori/tests/ -v
```

## Citation

If you use this software in your research, please cite:

```bibtex
@software{pcori_sitl_2026,
  title = {PCORI: Human-Centered AI for Clinical Decision Support},
  author = {Stony Brook University},
  year = {2026},
  url = {https://github.com/StonyBrookDB/PCORI}
}
```

## Acknowledgments

This project is funded by the [Patient-Centered Outcomes Research Institute (PCORI)](https://www.pcori.org/).

**Research Team**: Stony Brook University Department of Biomedical Informatics

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions about this project, please contact the research team at Stony Brook University.

---

*Disclaimer: This software is intended for research purposes. Clinical deployment requires appropriate validation and regulatory approval.*
