# System Architecture

This document describes the architecture of the PCORI Clinical Decision Support system.

## Overview

The system is designed as a modular, layered architecture that separates concerns across data processing, model training, explainability, and user interaction.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              User Layer                                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │  Clinicians     │  │  Researchers    │  │  Data Scientists│              │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘              │
│           │                    │                    │                        │
└───────────┼────────────────────┼────────────────────┼────────────────────────┘
            │                    │                    │
            ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Application Layer                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    SITL Dashboard (Web UI)                           │    │
│  │  • Cohort Builder    • Training UI      • Patient Explorer           │    │
│  │  • Model Results     • AI Chat          • Export Tools               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│  ┌─────────────────────────────────┴───────────────────────────────────┐    │
│  │                    FastAPI Backend (REST API)                        │    │
│  │  • Authentication    • Cohort Filtering  • Model Training            │    │
│  │  • Predictions       • Explanations      • Session Management        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ML Layer                                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │ Training Engine │  │ Model Registry  │  │ SHAP Engine     │              │
│  │ • LightGBM      │  │ • Model Storage │  │ • TreeExplainer │              │
│  │ • PyTorch       │  │ • Versioning    │  │ • DeepExplainer │              │
│  │ • Scikit-learn  │  │ • Metadata      │  │ • Summary Plots │              │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Data Layer                                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │ Feature Store   │  │ Patient DB      │  │ Sequence Cache  │              │
│  │ (Parquet)       │  │ (SQLite)        │  │ (NumPy)         │              │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘              │
│           ▲                    ▲                    ▲                        │
│           └────────────────────┴────────────────────┘                        │
│                                │                                             │
│  ┌─────────────────────────────┴───────────────────────────────────────┐    │
│  │                    Data Pipeline (ETL)                               │    │
│  │  • Raw EHR Ingestion    • Feature Engineering    • Validation        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. SITL Dashboard

The Stakeholder-in-the-Loop dashboard is the primary user interface for clinicians.

**Technology Stack:**
- Frontend: HTML5, JavaScript, Chart.js
- Backend: FastAPI (Python)
- Authentication: Cookie-based sessions

**Key Features:**
- **Cohort Builder**: Define patient populations using clinical criteria
- **Training UI**: One-click model training with progress monitoring
- **Patient Explorer**: Drill into individual risk predictions
- **AI Chat**: Natural language explanations (GPT/Claude integration)

### 2. Training Pipeline

The ML training pipeline supports multiple model architectures.

**Supported Models:**

| Model | Framework | Use Case |
|-------|-----------|----------|
| LightGBM | lightgbm | Tabular data, fast training |
| Logistic Regression | scikit-learn | Baseline, interpretability |
| Random Forest | scikit-learn | Feature importance |
| LSTM | PyTorch | Sequential patient data |
| GRU | PyTorch | Sequential data, faster |

**Training Flow:**
```
Raw Features → Preprocessing → Train/Val Split → Model Training → Evaluation → SHAP
```

### 3. Feature Selection Module

Multiple methods for identifying predictive features:

```
┌─────────────────────────────────────────────────────────────┐
│                 Feature Selection Ensemble                   │
│                                                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │   NTK    │ │ LightGBM │ │ Elastic  │ │Recurrence│       │
│  │ Inspired │ │ Feature  │ │   Net    │ │Enrichment│       │
│  │ Analysis │ │   Imp.   │ │          │ │          │       │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘       │
│       │            │            │            │              │
│       └────────────┴─────┬──────┴────────────┘              │
│                          │                                   │
│                          ▼                                   │
│              ┌───────────────────────┐                      │
│              │   Ensemble Voting     │                      │
│              │  (3+ method overlap)  │                      │
│              └───────────────────────┘                      │
└─────────────────────────────────────────────────────────────┘
```

### 4. Data Layer

**Data Storage Formats:**

| Format | Use Case | Characteristics |
|--------|----------|-----------------|
| Parquet | Feature Store | Columnar, compressed, fast queries |
| SQLite | Patient DB | Relational, portable, ACID |
| NumPy | Sequence Cache | Dense arrays, memory-mapped |

**Data Flow:**
```
EHR System → ETL Pipeline → Feature Store → ML Training → Model Predictions
                                ↓
                          Patient Database → Dashboard Queries
```

## Deployment Architecture

### Development
```
Local Machine
├── Python venv
├── SQLite database
└── Synthetic data
```

### Production
```
Server (bmidb0)
├── Conda environment
├── Health Facts data (Parquet)
├── GPU for deep learning
└── Uvicorn + FastAPI
```

## Security Considerations

1. **Authentication**: Session-based with configurable providers
2. **Data Access**: Role-based access to patient data
3. **Audit Logging**: All model training and predictions logged
4. **API Security**: Rate limiting, input validation

## Scalability

- **Horizontal**: Multiple dashboard instances behind load balancer
- **Vertical**: GPU acceleration for deep learning models
- **Data**: Parquet partitioning for large datasets

## External Dependencies

| Service | Purpose | Required |
|---------|---------|----------|
| OpenAI API | AI explanations | Optional |
| Anthropic API | AI explanations | Optional |
| Health Facts | EHR data | Production only |

## Future Architecture

Planned enhancements:
- Kubernetes deployment
- Model versioning with MLflow
- Real-time prediction API
- Federated learning support
