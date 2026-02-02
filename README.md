# PCORI: Human-Centered AI for Clinical Decision Support

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PCORI Funded](https://img.shields.io/badge/PCORI-Funded-green.svg)](https://www.pcori.org/)
[![CI](https://github.com/StonyBrookDB/PCORI/actions/workflows/ci.yml/badge.svg)](https://github.com/StonyBrookDB/PCORI/actions)

A machine learning infrastructure for **Opioid Use Disorder (OUD) and Overdose (OD) risk prediction**, developed under PCORI Contract 23C3 at Stony Brook University.

## Project Overview

This project addresses the gap between ML model development and clinical adoption through:

- **Transparent Models**: Interpretable predictions with SHAP-based explanations
- **Clinician-in-the-Loop Validation**: Interactive dashboards for clinical review
- **Scalable Pipeline**: Data processing for large EHR datasets
- **Multi-Model Support**: LightGBM, Random Forest, LSTM, GRU

## Program Management

| Field | Value |
|-------|-------|
| **Contract** | PCORI 23C3 |
| **PI** | Fusheng Wang, PhD |
| **Institution** | Stony Brook University |
| **Start Date** | February 1, 2025 |
| **Reporting** | Interim reports every 6 months |

### Key Milestones

| Milestone | Due Date | Status |
|-----------|----------|--------|
| Study Protocol Submission | 2025-03-01 | Complete |
| IRB Documentation | 2025-03-01 | Complete |
| Cohort Identification | 2025-05-15 | Complete |
| Initial Model Design | 2025-07-15 | Complete |
| Progress Report #1 | 2025-07-31 | Complete |
| Model Release 1 (GitHub) | 2025-08-31 | Complete |
| Interpretability Methods | 2025-10-31 | In Progress |

### Governance

- **AHDT Meetings**: Bi-monthly All Hands Design Team meetings
- **Quarterly Surveys**: Stakeholder engagement surveys
- **PCORI Oversight**: Reports via PCORI Online portal

## Research Compliance

### IRB Status

- **Protocol**: IRB2023-00456 (Stony Brook University IRB)
- **Status**: Approved with annual continuing review
- **Expiration**: March 2026 (renewal in progress)

### Data Governance

- **Data Source**: Cerner Health Facts under executed DUA
- **De-identification**: HIPAA Safe Harbor compliant
- **Access Control**: Role-based, audit-logged
- **PHI**: No direct identifiers in any dataset

## Evidence & Validation

### Datasets

| Dataset | Records | Features | Purpose |
|---------|---------|----------|---------|
| Health Facts | ~70M encounters | 500+ | Training/validation |
| Synthetic | 10,000 patients | 50 | Development/testing |

### Model Performance (Internal Validation)

| Model | AUROC | AUPRC | Accuracy |
|-------|-------|-------|----------|
| LightGBM | 0.82 | 0.61 | 0.78 |
| LSTM (T=10) | 0.79 | 0.58 | 0.75 |
| Logistic Reg. | 0.75 | 0.52 | 0.72 |

*External validation planned for Q2 2026.*

### Fairness Monitoring

- Subgroup analysis by demographics (age, gender, race/ethnicity)
- Calibration curves across groups
- Disparate impact assessment planned

## Limitations & Intended Use

### Intended Use
- Research and model development
- Educational purposes
- Stakeholder-in-the-loop validation prototyping

### Limitations
- **Not for clinical decisions**: Models not validated for direct patient care
- **Population specificity**: Trained on specific populations
- **No FDA clearance**: Requires separate regulatory review for deployment

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    PCORI ML Infrastructure                    │
├──────────────────────────────────────────────────────────────┤
│  Raw EHR Data → Pipeline (ETL) → Feature Store (Parquet)     │
│                        ↓                                      │
│  Model Training: LightGBM | LSTM | GRU | Logistic Reg.       │
│                        ↓                                      │
│  Explainability: SHAP values, feature importance             │
│                        ↓                                      │
│  SITL Dashboard: Cohort Builder | Training UI | AI Chat      │
└──────────────────────────────────────────────────────────────┘
```

## Components

| Component | Description | Docs |
|-----------|-------------|------|
| [SITL Dashboard](SITL_Dashboard_PCORI/) | Clinician validation interface | [README](SITL_Dashboard_PCORI/README.md) |
| [Pipeline](pipeline-pcori/) | Training pipeline | [README](pipeline-pcori/README.md) |
| [Feature Selection](feature_selection/) | NTK, LightGBM, Elastic Net methods | [Docs](docs/FEATURE_SELECTION.md) |

## Quick Start

```bash
# Clone and setup
git clone https://github.com/StonyBrookDB/PCORI.git
cd PCORI
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run dashboard
cd SITL_Dashboard_PCORI
pip install -r requirements.txt
python -m uvicorn backend.main:app --port 8503

# Train model
cd ../pipeline-pcori
python train.py --dataset ./data/synth --model lightgbm
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Installation](docs/INSTALLATION.md)
- [API Reference](docs/API.md)
- [Data Pipeline](docs/DATA_PIPELINE.md)
- [Models](docs/MODELS.md)
- [Contributing](CONTRIBUTING.md)

## Citation

```bibtex
@software{pcori_sitl_2026,
  title = {PCORI: Human-Centered AI for Clinical Decision Support},
  author = {Wang, Fusheng and Liu, Yinan and Ding, Zihan},
  year = {2026},
  publisher = {Stony Brook University},
  url = {https://github.com/StonyBrookDB/PCORI},
  note = {Funded by PCORI Contract 23C3}
}
```

## Acknowledgments

**Funding**: Patient-Centered Outcomes Research Institute (PCORI) Contract 23C3

**Research Team**: Department of Biomedical Informatics, Stony Brook University

**Data Partner**: Cerner Corporation (Health Facts)

## License

MIT License - see [LICENSE](LICENSE)

---

*This software is for research purposes only. Clinical deployment requires regulatory approval.*
