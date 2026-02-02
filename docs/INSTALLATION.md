# Installation Guide

This guide covers setting up the PCORI Clinical Decision Support system.

## Prerequisites

- Python 3.8 or higher
- pip or conda package manager
- Git
- (Optional) CUDA-capable GPU for deep learning models

## Quick Installation

### 1. Clone the Repository

```bash
git clone https://github.com/StonyBrookDB/PCORI.git
cd PCORI
```

### 2. Create Virtual Environment

**Using venv:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

**Using conda:**
```bash
conda create -n pcori python=3.10
conda activate pcori
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Component-Specific Setup

### SITL Dashboard

```bash
cd SITL_Dashboard_PCORI
pip install -r requirements.txt

# Initialize database (synthetic data)
python backend/init_db.py

# Start server
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8503
```

### Training Pipeline

```bash
cd pipeline-pcori
pip install -r requirements.txt

# Verify installation
python train.py --help
```

## Environment Variables

Create a `.env` file in the project root:

```bash
# Data paths
DATA_ROOT=/path/to/data

# Optional: AI chat features
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=...

# Optional: Database
DATABASE_URL=sqlite:///./data/patients.db
```

## Verifying Installation

```bash
# Run tests
pytest tests/ -v

# Check dashboard
curl http://localhost:8503/health
```

## Troubleshooting

### Common Issues

**ImportError: No module named 'xxx'**
```bash
pip install xxx
```

**CUDA out of memory**
- Reduce batch size in training configuration
- Use CPU-only mode: `--device cpu`

**Port already in use**
```bash
python -m uvicorn backend.main:app --port 8504
```

## Next Steps

- [Architecture Overview](ARCHITECTURE.md)
- [Data Pipeline Guide](DATA_PIPELINE.md)
- [API Reference](API.md)
