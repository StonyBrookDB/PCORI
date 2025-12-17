"""Pydantic request/response schemas for the patient story service."""

from typing import List, Optional, Literal

from pydantic import BaseModel, Field, ConfigDict


class Medication(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    dose: str
    frequency: str


class Lab(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    value: float
    unit: str


class Encounter(BaseModel):
    model_config = ConfigDict(extra="forbid")
    date: str
    type: Literal["outpatient", "inpatient", "emergency", "followup", "telehealth", "home_health"]
    diagnoses: List[str]
    medications: List[Medication] = Field(default_factory=list)
    labs: List[Lab] = Field(default_factory=list)
    notes: Optional[str] = None


class OpioidRisk(BaseModel):
    model_config = ConfigDict(extra="forbid")
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_level: Literal["low", "medium", "high"]
    risk_factors: List[str] = Field(default_factory=list)


class PatientStoryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patient_id: str = Field(min_length=1)
    age: int = Field(gt=0, le=120)
    sex: str
    encounters: List[Encounter]
    opioid_risk: OpioidRisk
    language: str = "en"
    prompt_version: str = "v1"
    schema_version: str = "v1"


class PatientStoryResponse(BaseModel):
    patient_id: str
    fingerprint: str
    model_name: str
    prompt_version: str
    generated_at: str
    clinician_summary: str
    patient_story: str
    disclaimer: str
