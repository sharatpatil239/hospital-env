from __future__ import annotations

from typing import Dict, List, Set, Tuple

from .models import Patient, Symptom, TreatmentType, UrgencyLevel


SYMPTOM_TREATMENT_MAP: Dict[Symptom, Tuple[Set[TreatmentType], Set[TreatmentType]]] = {
    Symptom.CHEST_PAIN: ({TreatmentType.CARDIAC_INTERVENTION}, {TreatmentType.ICU_MONITORING}),
    Symptom.SHORTNESS_OF_BREATH: ({TreatmentType.RESPIRATORY_SUPPORT}, {TreatmentType.ICU_MONITORING}),
    Symptom.HIGH_FEVER: ({TreatmentType.ANTIBIOTICS}, {TreatmentType.OBSERVATION}),
    Symptom.TRAUMA: ({TreatmentType.EMERGENCY_SURGERY}, {TreatmentType.ICU_MONITORING}),
    Symptom.STROKE_SYMPTOMS: ({TreatmentType.NEUROLOGICAL_EVAL}, {TreatmentType.ICU_MONITORING}),
    Symptom.FRACTURE: ({TreatmentType.ORTHOPEDIC_CARE}, {TreatmentType.PAIN_MANAGEMENT}),
    Symptom.MILD_INFECTION: ({TreatmentType.ANTIBIOTICS}, {TreatmentType.OBSERVATION}),
    Symptom.HEADACHE: ({TreatmentType.PAIN_MANAGEMENT}, {TreatmentType.OBSERVATION}),
    Symptom.NAUSEA: ({TreatmentType.OBSERVATION}, {TreatmentType.DISCHARGE}),
}

ICU_REQUIRED_SYMPTOMS = {
    Symptom.CHEST_PAIN,
    Symptom.SHORTNESS_OF_BREATH,
    Symptom.STROKE_SYMPTOMS,
    Symptom.TRAUMA,
}

URGENCY_SCORE = {
    UrgencyLevel.CRITICAL: 4,
    UrgencyLevel.HIGH: 3,
    UrgencyLevel.MODERATE: 2,
    UrgencyLevel.LOW: 1,
}


# 🔥 IMPROVED: multi-symptom aware scoring
def treatment_score(patient: Patient, treatment: TreatmentType) -> float:
    scores = []

    for symptom in patient.symptoms:
        correct, acceptable = SYMPTOM_TREATMENT_MAP.get(symptom, (set(), set()))

        if treatment in correct:
            scores.append(1.0)
        elif treatment in acceptable:
            scores.append(0.5)
        else:
            scores.append(0.0)

    base_score = max(scores) if scores else 0.0

    # harmful discharge
    if treatment == TreatmentType.DISCHARGE and patient.severity > 0.6:
        return -1.0

    # overtreatment
    if patient.severity < 0.4 and treatment in {
        TreatmentType.EMERGENCY_SURGERY,
        TreatmentType.ICU_MONITORING,
    }:
        return -0.5

    return base_score


def needs_icu(patient: Patient) -> bool:
    if patient.severity >= 0.7:
        return True
    if any(s in ICU_REQUIRED_SYMPTOMS for s in patient.symptoms):
        return patient.severity >= 0.5
    return False


def optimal_priority_order(patients: List[Patient]) -> List[str]:
    return [
        p.patient_id
        for p in sorted(
            patients,
            key=lambda p: (-URGENCY_SCORE[p.urgency], -p.severity, -p.hours_waiting),
        )
    ]


# 🔥 stricter scoring
def priority_rank_score(patient_id: str, assigned_rank: int, optimal_order: List[str]) -> float:
    n = len(optimal_order)
    if n == 0:
        return 1.0

    try:
        optimal_pos = optimal_order.index(patient_id) + 1
    except ValueError:
        return 0.0

    deviation = abs(optimal_pos - assigned_rank)

    # harsher penalty
    return max(0.0, 1.0 - (deviation / n) ** 1.5)


def compute_wait_penalty(hours_waiting: float, threshold=2.0, rate=0.07) -> float:
    if hours_waiting <= threshold:
        return 0.0
    return max(-0.5, -(hours_waiting - threshold) * rate)