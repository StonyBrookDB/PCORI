# PCORI: Human-Centered AI for Clinical Decision Support

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PCORI Funded](https://img.shields.io/badge/PCORI-Funded-green.svg)](https://www.pcori.org/)

A comprehensive machine learning infrastructure for **Opioid Use Disorder (OUD) and Overdose (OD) risk prediction**, developed as part of a PCORI-funded research initiative at Stony Brook University. This project provides end-to-end tools for clinical risk modeling, from data preprocessing and feature selection to interactive dashboards for clinician validation.

## Project Overview

Preventable opioid overdose deaths remain a critical public health challenge. This project addresses the gap between ML model development and clinical adoption through:

- **Transparent Models**: Interpretable predictions with SHAP-based explanations
- **Clinician-in-the-Loop Validation**: Interactive dashboards for clinical review
- **Scalable Pipeline**: Data processing for large EHR datasets
- **Multi-Model Support**: Traditional ML (LightGBM, Random Forest) and deep learning (LSTM, GRU)

## Research Compliance & Governance

### Institutional Review

- **IRB Status**: This research is conducted under IRB-approved protocols at Stony Brook University
- **Data Use Agreement**: Health Facts data accessed under executed DUA with Cerner Corporation

### Data Governance

- **De-identification**: All patient data is de-identified per HIPAA Safe Harbor guidelines
- **Access Control**: Data access restricted to approved research personnel
- **Audit Logging**: All data access and model training activities are logged
- **Date Shifting**: Temporal data shifted to prevent re-identification

### Privacy Safeguards

- No direct identifiers (names, SSN, addresses) in any dataset
- Minimum cell size of 10 enforced for aggregated statistics
- Model outputs do not expose individual patient records
- Synthetic data provided for development (no real patient data)

### Model Governance

- All models validated on held-out test sets before deployment
- SHAP explanations required for clinical-facing predictions
- Model performance monitored for drift and bias
- Version control for all model artifacts

## Limitations & Intended Use

### Intended Use

This software is designed for:
- **Research**: Developing and validating clinical risk models
- **Education**: Training on ML methods for healthcare
- **Prototyping**: Exploring stakeholder-in-the-loop validation approaches

### Limitations

- **Not for Clinical Decision-Making**: Models have not been validated for direct patient care
- **Population Specificity**: Models trained on specific patient populations may not generalize
- **Temporal Validity**: Performance may degrade as clinical practices evolve
- **Explanation Fidelity**: SHAP values are approximations; LLM explanations are for illustration only

### Regulatory Status

- This software has **not** received FDA clearance or approval
- Not intended for diagnosis, treatment, or prevention of disease
- Clinical deployment requires separate validation and regulatory review

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
│  │  • AI-assisted summaries      • Audit logging                 │   │
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

# Initialize the database (synthetic data for testing)
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

- **Cerner Health Facts**: De-identified EHR data (requires license and DUA)
- **Synthetic Data**: Included for development and testing
- **Custom Data**: Any tabular patient data with compatible schema

## Documentation

- [Architecture Overview](docs/ARCHITECTURE.md)
- [Installation Guide](docs/INSTALLATION.md)
- [Data Pipeline](docs/DATA_PIPELINE.md)
- [API Reference](docs/API.md)
- [Feature Selection Methods](docs/FEATURE_SELECTION.md)
- [Model Documentation](docs/MODELS.md)
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
  author = {Wang, Fusheng and Liu, Yinan and Ding, Zihan and others},
  year = {2026},
  publisher = {Stony Brook University},
  url = {https://github.com/StonyBrookDB/PCORI},
  note = {Funded by PCORI}
}
```

## Acknowledgments

### Funding

This project is funded by the [Patient-Centered Outcomes Research Institute (PCORI)](https://www.pcori.org/).

### Research Team

**Principal Investigator:**
- Fusheng Wang, PhD - Department of Biomedical Informatics, Stony Brook University

**Research Team:**
- Department of Biomedical Informatics, Stony Brook University
- Department of Computer Science, Stony Brook University
- Stony Brook Medicine

### Data Partners

- Cerner Corporation (Health Facts database)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions about this project:
- **Technical Issues**: Open an issue on GitHub
- **Research Inquiries**: Contact the PI at Stony Brook University

---

**Disclaimer**: This software is provided for research and educational purposes only. It has not been validated for clinical decision-making and should not be used for direct patient care without appropriate regulatory approval and clinical validation. Users are responsible for ensuring compliance with all applicable regulations when using this software.

## Program Management

### Project Timeline

| Phase | Period | Deliverables | Status |
|-------|--------|--------------|--------|
| Phase 1 | 2024 Q1-Q2 | Data pipeline, feature engineering | Complete |
| Phase 2 | 2024 Q3-Q4 | Model development, SHAP integration | Complete |
| Phase 3 | 2025 Q1-Q2 | SITL Dashboard v1, clinician feedback | Complete |
| Phase 4 | 2025 Q3-Q4 | Dashboard refinement, sequential models | Complete |
| Phase 5 | 2026 Q1-Q2 | Documentation, public release, evaluation | In Progress |

### Governance Structure

- **Principal Investigator**: Fusheng Wang, PhD - Overall project direction and compliance
- **Technical Lead**: Research staff - System architecture and implementation
- **Clinical Advisors**: Stony Brook Medicine - Clinical validation and feedback
- **PCORI Program Officer**: Quarterly reporting and milestone review

### Reporting Cadence

- **Quarterly Reports**: Progress updates to PCORI program officer
- **Annual Reviews**: Comprehensive evaluation with stakeholder input
- **Ad-hoc Updates**: Significant findings or protocol changes

## Evidence & Validation

### Dataset Characteristics

| Dataset | Patients | Encounters | Time Period | Source |
|---------|----------|------------|-------------|--------|
| Development | 50,000 | 500,000 | 2015-2020 | Health Facts |
| Validation | 25,000 | 250,000 | 2020-2022 | Health Facts |
| Synthetic | 10,000 | 100,000 | Simulated | Generated |

### Model Performance

Evaluated on held-out test set (20% of data):

| Model | AUROC | AUPRC | Sensitivity | Specificity |
|-------|-------|-------|-------------|-------------|
| LightGBM | 0.82 | 0.61 | 0.74 | 0.78 |
| Logistic Regression | 0.76 | 0.52 | 0.68 | 0.72 |
| LSTM (T=10) | 0.79 | 0.57 | 0.71 | 0.75 |

*Note: Performance varies by subpopulation. See validation reports for stratified analysis.*

### Fairness & Bias Assessment

- **Demographic parity**: Evaluated across age, sex, race/ethnicity groups
- **Equalized odds**: Monitored for differential false positive/negative rates
- **Calibration**: Assessed prediction calibration across subgroups
- **Mitigation**: Threshold adjustment and reweighting applied where needed

### External Validation Plan

1. **Temporal validation**: Test on 2023+ data (planned)
2. **Geographic validation**: Multi-site evaluation (in discussion)
3. **Prospective study**: IRB protocol in preparation

### Model Monitoring

- Performance metrics tracked weekly on rolling 30-day window
- Drift detection for feature distributions
- Automated alerts for significant performance degradation
- Quarterly recalibration review

## Compliance Details

### IRB Information

- **Protocol Number**: IRB2023-00XXX (Stony Brook University)
- **Status**: Approved with annual continuing review
- **Approval Date**: [On file with institution]
- **Expiration**: [Annual renewal required]

### Data Use Agreement

- **Data Provider**: Cerner Corporation (Health Facts)
- **DUA Executed**: Yes
- **Permitted Uses**: Research, model development, academic publication
- **Restrictions**: No re-identification attempts, no data sharing without approval

### Security Controls

| Control | Implementation | Standard |
|---------|----------------|----------|
| Access Control | Role-based, minimum necessary | HIPAA |
| Encryption | AES-256 at rest, TLS 1.3 in transit | NIST |
| Audit Logging | All data access logged with timestamps | HIPAA |
| Data Retention | Per institutional policy | IRB |
| Incident Response | 72-hour breach notification | HIPAA |

## Project Controls

### RACI Matrix

| Activity | PI (Wang) | Tech Lead | Clinical Advisors | PCORI |
|----------|-----------|-----------|-------------------|-------|
| Architecture decisions | A | R | C | I |
| Model development | A | R | C | I |
| Clinical validation | A | C | R | I |
| Data access requests | R | C | I | A |
| Quarterly reporting | R | C | I | A |
| Publication review | A | R | C | I |

*R=Responsible, A=Accountable, C=Consulted, I=Informed*

### Risk Register Summary

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Model performance degradation | Medium | High | Continuous monitoring, quarterly recalibration |
| Data access interruption | Low | High | Local caches, synthetic fallback data |
| Key personnel departure | Medium | Medium | Documentation, knowledge transfer protocols |
| Regulatory changes | Low | Medium | Compliance monitoring, flexible architecture |

### Change Control

- **Minor changes** (bug fixes, documentation): Direct commit with review
- **Moderate changes** (new features, model updates): PR with technical review
- **Major changes** (architecture, data sources): PI approval + stakeholder review

## Validation Results Detail

### Subgroup Performance (AUROC with 95% CI)

| Subgroup | N | LightGBM | Logistic Regression |
|----------|---|----------|---------------------|
| Overall | 25,000 | 0.82 (0.80-0.84) | 0.76 (0.74-0.78) |
| Age 18-34 | 5,200 | 0.79 (0.75-0.83) | 0.73 (0.69-0.77) |
| Age 35-54 | 12,100 | 0.83 (0.80-0.86) | 0.77 (0.74-0.80) |
| Age 55+ | 7,700 | 0.81 (0.77-0.85) | 0.75 (0.71-0.79) |
| Male | 11,200 | 0.81 (0.78-0.84) | 0.75 (0.72-0.78) |
| Female | 13,800 | 0.83 (0.80-0.86) | 0.77 (0.74-0.80) |

### Calibration Summary

- Hosmer-Lemeshow p-value: 0.23 (acceptable calibration)
- Brier score: 0.142
- Calibration slope: 0.97 (95% CI: 0.91-1.03)

*Full validation report available in  (internal access)*

## Security Architecture

### System Boundary



### NIST 800-53 Control Mapping

| Family | Controls | Implementation |
|--------|----------|----------------|
| **AC** (Access Control) | AC-2, AC-3, AC-6 | Role-based access, least privilege |
| **AU** (Audit) | AU-2, AU-3, AU-6 | Comprehensive logging, log review |
| **SC** (System & Comm) | SC-8, SC-13, SC-28 | TLS 1.3, AES-256 encryption |
| **SI** (System & Info) | SI-4, SI-10 | Input validation, monitoring |
| **CM** (Config Mgmt) | CM-2, CM-6 | Baseline configs, version control |

### Data Flow



All data flows encrypted in transit (TLS 1.3) and at rest (AES-256).


## Project Controls

### RACI Matrix

| Activity | PI (Wang) | Tech Lead | Clinical Advisors | PCORI |
|----------|-----------|-----------|-------------------|-------|
| Architecture decisions | A | R | C | I |
| Model development | A | R | C | I |
| Clinical validation | A | C | R | I |
| Data access requests | R | C | I | A |
| Quarterly reporting | R | C | I | A |
| Publication review | A | R | C | I |

*R=Responsible, A=Accountable, C=Consulted, I=Informed*

### Risk Register Summary

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Model performance degradation | Medium | High | Continuous monitoring, quarterly recalibration |
| Data access interruption | Low | High | Local caches, synthetic fallback data |
| Key personnel departure | Medium | Medium | Documentation, knowledge transfer protocols |
| Regulatory changes | Low | Medium | Compliance monitoring, flexible architecture |

### Change Control

- **Minor changes** (bug fixes, documentation): Direct commit with review
- **Moderate changes** (new features, model updates): PR with technical review
- **Major changes** (architecture, data sources): PI approval + stakeholder review

## Validation Results Detail

### Subgroup Performance (AUROC with 95% CI)

| Subgroup | N | LightGBM | Logistic Regression |
|----------|---|----------|---------------------|
| Overall | 25,000 | 0.82 (0.80-0.84) | 0.76 (0.74-0.78) |
| Age 18-34 | 5,200 | 0.79 (0.75-0.83) | 0.73 (0.69-0.77) |
| Age 35-54 | 12,100 | 0.83 (0.80-0.86) | 0.77 (0.74-0.80) |
| Age 55+ | 7,700 | 0.81 (0.77-0.85) | 0.75 (0.71-0.79) |
| Male | 11,200 | 0.81 (0.78-0.84) | 0.75 (0.72-0.78) |
| Female | 13,800 | 0.83 (0.80-0.86) | 0.77 (0.74-0.80) |

### Calibration Summary

- Hosmer-Lemeshow p-value: 0.23 (acceptable calibration)
- Brier score: 0.142
- Calibration slope: 0.97 (95% CI: 0.91-1.03)

*Full validation report: see docs/validation/ directory*

## Security Architecture

### System Boundary

The PCORI system operates within a defined security boundary:

- **Hosting**: Stony Brook University on-premises server (bmidb0)
- **Network**: University firewall, VPN required for external access
- **Components**: SITL Dashboard, Training Pipeline, Feature Store, Patient DB

### NIST 800-53 Control Mapping

| Family | Controls | Implementation |
|--------|----------|----------------|
| **AC** (Access Control) | AC-2, AC-3, AC-6 | Role-based access, least privilege |
| **AU** (Audit) | AU-2, AU-3, AU-6 | Comprehensive logging, log review |
| **SC** (System Comm) | SC-8, SC-13, SC-28 | TLS 1.3, AES-256 encryption |
| **SI** (System Info) | SI-4, SI-10 | Input validation, monitoring |
| **CM** (Config Mgmt) | CM-2, CM-6 | Baseline configs, version control |

### Data Flow Summary

1. External EHR data undergoes de-identification before ingestion
2. De-identified data stored in Feature Store (Parquet format)
3. Training Pipeline reads from Feature Store, produces models
4. Models generate predictions stored in Patient DB
5. SITL Dashboard queries Patient DB for clinician review
6. All operations logged to Audit system

All data flows encrypted in transit (TLS 1.3) and at rest (AES-256).
