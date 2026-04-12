from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ─────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────

class Symptom(str, Enum):
    CHEST_PAIN = "chest_pain"
    SHORTNESS_OF_BREATH = "shortness_of_breath"
    HIGH_FEVER = "high_fever"
    TRAUMA = "trauma"
    STROKE_SYMPTOMS = "stroke_symptoms"
    ABDOMINAL_PAIN = "abdominal_pain"
    FRACTURE = "fracture"
    MILD_INFECTION = "mild_infection"
    HEADACHE = "headache"
    NAUSEA = "nausea"


class TreatmentType(str, Enum):
    EMERGENCY_SURGERY = "emergency_surgery"
    CARDIAC_INTERVENTION = "cardiac_intervention"
    ICU_MONITORING = "icu_monitoring"
    ANTIBIOTICS = "antibiotics"
    PAIN_MANAGEMENT = "pain_management"
    RESPIRATORY_SUPPORT = "respiratory_support"
    NEUROLOGICAL_EVAL = "neurological_eval"
    ORTHOPEDIC_CARE = "orthopedic_care"
    OBSERVATION = "observation"
    DISCHARGE = "discharge"


class UrgencyLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"


# ─────────────────────────────────────────────
# PATIENT
# ─────────────────────────────────────────────

class Patient(BaseModel):
    patient_id: str
    symptoms: List[Symptom]
    severity: float = Field(..., ge=0.0, le=1.0)
    urgency: UrgencyLevel
    age: int = Field(..., ge=0, le=120)
    comorbidities: List[str] = Field(default_factory=list)
    hours_waiting: float = Field(default=0.0, ge=0.0)

    @field_validator("severity")
    @classmethod
    def round_severity(cls, v: float) -> float:
        return round(v, 4)


# ─────────────────────────────────────────────
# RESOURCES
# ─────────────────────────────────────────────

class HospitalResources(BaseModel):
    icu_beds_available: int = Field(..., ge=0)
    icu_beds_total: int = Field(..., ge=0)
    general_beds_available: int = Field(..., ge=0)
    general_beds_total: int = Field(..., ge=0)
    doctors_available: int = Field(..., ge=0)
    doctors_total: int = Field(..., ge=0)

    @model_validator(mode="after")
    def validate_capacity(self):
        if self.icu_beds_available > self.icu_beds_total:
            raise ValueError("ICU available > total")
        if self.general_beds_available > self.general_beds_total:
            raise ValueError("General beds available > total")
        if self.doctors_available > self.doctors_total:
            raise ValueError("Doctors available > total")
        return self

    @property
    def icu_utilization(self) -> float:
        if self.icu_beds_total == 0:
            return 0.0
        return round(1.0 - (self.icu_beds_available / self.icu_beds_total), 4)

    @property
    def general_utilization(self) -> float:
        if self.general_beds_total == 0:
            return 0.0
        return round(1.0 - (self.general_beds_available / self.general_beds_total), 4)


# ─────────────────────────────────────────────
# OBSERVATION
# ─────────────────────────────────────────────

class Observation(BaseModel):
    step: int = Field(default=0, ge=0)
    patients: List[Patient]
    resources: HospitalResources
    pending_patients: int = Field(default=0, ge=0)
    time_elapsed_hours: float = Field(default=0.0, ge=0.0)
    task_id: str
    task_description: str


# ─────────────────────────────────────────────
# ACTION
# ─────────────────────────────────────────────

class PatientAction(BaseModel):
    patient_id: str
    treatment: TreatmentType
    assign_icu_bed: bool = False
    assign_general_bed: bool = False
    assign_doctor: bool = True
    priority_rank: int = Field(..., ge=1)

    @model_validator(mode="after")
    def validate_bed_assignment(self):
        # ❌ Prevent unrealistic assignment
        if self.assign_icu_bed and self.assign_general_bed:
            raise ValueError("Cannot assign both ICU and General bed to same patient")
        return self


class Action(BaseModel):
    patient_actions: List[PatientAction]
    reasoning: Optional[str] = None

    @field_validator("patient_actions")
    @classmethod
    def unique_patients(cls, v):
        ids = [a.patient_id for a in v]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate patient_id in actions")
        return v

    @field_validator("patient_actions")
    @classmethod
    def unique_priority(cls, v):
        ranks = [a.priority_rank for a in v]
        if len(ranks) != len(set(ranks)):
            raise ValueError("Priority ranks must be unique")
        return v


# ─────────────────────────────────────────────
# REWARD
# ─────────────────────────────────────────────

class RewardBreakdown(BaseModel):
    treatment_correctness: float = 0.0
    priority_correctness: float = 0.0
    resource_efficiency: float = 0.0
    critical_patient_bonus: float = 0.0
    resource_misuse_penalty: float = 0.0
    ignored_critical_penalty: float = 0.0
    wait_time_penalty: float = 0.0
    raw_total: float = 0.0


class Reward(BaseModel):
    value: float
    normalized: float
    breakdown: RewardBreakdown

    @model_validator(mode="after")
    def clamp(self):
        self.value = round(max(-1.0, min(1.0, self.value)), 6)
        self.normalized = round((self.value + 1.0) / 2.0, 6)
        return self