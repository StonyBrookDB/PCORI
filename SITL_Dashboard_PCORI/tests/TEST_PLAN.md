# SITL Dashboard Test Plan

## Overview
This document outlines the test plan for the SITL Dashboard, covering database persistence, API endpoints, training jobs, and UI functionality.

---

## Test Categories

### 1. Database Tests
- [ ] Database initialization creates tables (jobs, models)
- [ ] Jobs table CRUD operations work
- [ ] Models table CRUD operations work
- [ ] Migration from jobs to models works correctly
- [ ] Indexes are created for filtering

### 2. API Endpoint Tests

#### Models API
- [ ] `GET /api/models` returns all models
- [ ] `GET /api/models?dataset_id=X` filters by dataset
- [ ] `GET /api/models?search=X` filters by search query
- [ ] `GET /api/models/{id}` returns specific model
- [ ] Model data includes all required fields

#### Jobs API
- [ ] `GET /api/jobs` returns all jobs
- [ ] `GET /api/jobs/{id}/status` returns job status
- [ ] `GET /api/jobs/{id}/log` returns job logs
- [ ] `GET /api/jobs/{id}/training_curves` returns curves data

#### Training API
- [ ] `POST /api/train/submit` creates a job
- [ ] Job status transitions: queued → running → completed
- [ ] Completed job creates model in database
- [ ] Failed job records error message

### 3. Training Tests
- [ ] LightGBM training completes successfully
- [ ] Logistic Regression training completes successfully
- [ ] PyTorch MLP training completes successfully
- [ ] Hyperparameters are passed correctly
- [ ] Metrics are saved to final_metrics.json
- [ ] Training history is recorded

### 4. Integration Tests
- [ ] Server starts without errors
- [ ] Models survive server restart (persistence)
- [ ] Dataset filtering works end-to-end
- [ ] Search filtering works end-to-end

### 5. UI Tests (Manual)
- [ ] Training tab shows hyperparameter inputs
- [ ] Model Results tab shows filtered models
- [ ] Training Status tab shows jobs
- [ ] Search input filters in real-time
- [ ] "This dataset only" checkbox works

---

## Test Execution

### Automated Tests
Run with:
```bash
cd /home/yinan/checkouts/v0/apps/sitl_dashboard
PYTHONPATH="/home/yinan/checkouts/v0/packages" \
conda run -n eliu3 python -m pytest tests/ -v
```

### Manual Tests
1. Start server
2. Open browser to http://localhost:8765/
3. Follow UI test checklist above

---

## Test Data

### Datasets Available
- `synthetic` - 2000 samples, classification
- `breast_cancer` - 569 samples, classification
- `diabetes` - 442 samples, regression

### Expected Model Fields
```python
{
    "model_id": str,
    "job_id": str,
    "dataset_id": str,
    "model_type": str,  # lightgbm, logreg, pytorch
    "name": str,
    "task_type": str,  # classification, regression
    "features": List[str],
    "train_metrics": Dict,
    "val_metrics": Dict,
    "feature_importance": List,
    "training_history": Dict,
    "model_path": str,
    "work_dir": str,
    "created_at": str,  # ISO format
}
```
