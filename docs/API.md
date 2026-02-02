# API Reference

This document describes the REST API endpoints for the SITL Dashboard.

## Base URL

```
http://localhost:8503/api
```

## Authentication

### Login

```http
POST /api/login
Content-Type: application/x-www-form-urlencoded

username=demo&password=demo
```

**Response:** Sets session cookie

### Logout

```http
POST /api/logout
```

## Cohort Endpoints

### Filter Patients

```http
POST /api/cohort/filter
Content-Type: application/json

{
  "age_min": 18,
  "age_max": 65,
  "has_opioid_rx": true,
  "diagnosis_codes": ["F11.1", "F11.2"]
}
```

**Response:**
```json
{
  "patient_ids": ["P001", "P002", "P003"],
  "count": 3,
  "summary": {
    "age_mean": 42.5,
    "male_pct": 0.67
  }
}
```

## Training Endpoints

### Quick Train (LightGBM/LogReg)

```http
POST /api/train/quick
Content-Type: application/json

{
  "model_type": "lightgbm",
  "patient_ids": ["P001", "P002"],
  "features": ["age", "rx_count", "prior_od"]
}
```

**Response:**
```json
{
  "model_id": "m_abc123",
  "metrics": {
    "auroc": 0.823,
    "auprc": 0.612,
    "accuracy": 0.784
  },
  "feature_importance": {
    "prior_od": 0.42,
    "rx_count": 0.31,
    "age": 0.15
  },
  "training_time_ms": 287
}
```

### Sequential Train (LSTM/GRU)

```http
POST /api/dataset/health_facts_seq/train
Content-Type: application/json

{
  "model_type": "lstm",
  "T": 10,
  "feature_set": "core",
  "hidden_size": 128,
  "epochs": 20
}
```

**Response:**
```json
{
  "job_id": "job_xyz789",
  "status": "running"
}
```

### Get Training Progress

```http
GET /api/dataset/health_facts_seq/training/{job_id}
```

**Response:**
```json
{
  "status": "running",
  "progress": 0.45,
  "current_epoch": 9,
  "metrics": {
    "train_loss": 0.342,
    "val_auroc": 0.756
  }
}
```

## Model Endpoints

### List Models

```http
GET /api/models
```

### Get Model Details

```http
GET /api/model/{model_id}
```

### Get Patient Explanation

```http
GET /api/model/{model_id}/explain/patient/{patient_id}
```

**Response:**
```json
{
  "patient_id": "P001",
  "risk_score": 0.73,
  "risk_level": "High",
  "shap_values": {
    "prior_od": 0.28,
    "rx_count": 0.19,
    "age": -0.05
  }
}
```

## AI Chat Endpoint

### Get AI Explanation

```http
POST /api/chat
Content-Type: application/json

{
  "message": "Explain this patient's risk factors",
  "context": {
    "patient_id": "P001",
    "model_id": "m_abc123"
  }
}
```

**Response:**
```json
{
  "response": "This patient has elevated risk primarily due to...",
  "disclaimer": "AI-generated explanation for research purposes only."
}
```

## Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

## Error Responses

All endpoints return errors in this format:

```json
{
  "error": true,
  "message": "Description of the error",
  "code": "ERROR_CODE"
}
```

| Code | HTTP Status | Description |
|------|-------------|-------------|
| AUTH_REQUIRED | 401 | Authentication required |
| NOT_FOUND | 404 | Resource not found |
| VALIDATION_ERROR | 422 | Invalid request data |
| SERVER_ERROR | 500 | Internal server error |

## Rate Limits

- 100 requests per minute per session
- Training endpoints: 10 concurrent jobs max
