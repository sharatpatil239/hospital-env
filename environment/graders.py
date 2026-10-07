from __future__ import annotations

from typing import Dict, List, Tuple

from .models import (
    Action,
    HospitalResources,
    Observation,
    Patient,
    UrgencyLevel,
)
from .utils import (
    needs_icu,
    optimal_priority_order,
    priority_rank_score,
    treatment_score,
)


# ─────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────

class GradeResult:
    def __init__(self, score: float, details: Dict[str, float]):
        self.score   = round(max(0.0, min(1.0, score)), 4)
        self.details = {k: round(v, 4) for k, v in details.items()}

    def __repr__(self) -> str:
        return f"GradeResult(score={self.score}, details={self.details})"


# ─────────────────────────────────────────────
# Treatment scoring
# ─────────────────────────────────────────────

def _score_treatments(patients: List[Patient], action: Action) -> float:
    action_map = {pa.patient_id: pa for pa in action.patient_actions}
    weighted_sum = 0.0
    weight_total = 0.0

    for patient in patients:
        weight = 0.5 + 0.5 * patient.severity
        pa = action_map.get(patient.patient_id)

        if pa is None:
            is_urgent = patient.urgency in (UrgencyLevel.CRITICAL, UrgencyLevel.HIGH)
            raw = -1.0 if is_urgent else -0.3
        else:
            raw = treatment_score(patient, pa.treatment)

        normalized = (raw + 1.0) / 2.0
        weighted_sum += weight * normalized
        weight_total += weight

    return weighted_sum / weight_total if weight_total > 0 else 0.0


# ─────────────────────────────────────────────
# Priority scoring
# ─────────────────────────────────────────────

def _score_priorities(patients: List[Patient], action: Action) -> float:
    optimal_order = optimal_priority_order(patients)
    action_map = {pa.patient_id: pa for pa in action.patient_actions}

    scores = []
    for patient in patients:
        pa = action_map.get(patient.patient_id)
        if pa is None:
            scores.append(0.0)
        else:
            scores.append(
                priority_rank_score(
                    patient.patient_id, pa.priority_rank, optimal_order
                )
            )

    return sum(scores) / len(scores) if scores else 0.0


# ─────────────────────────────────────────────
# Resource scoring (STRICT VERSION)
# ─────────────────────────────────────────────

def _score_resources(
    patients: List[Patient],
    action: Action,
    resources: HospitalResources,
) -> Tuple[float, float]:

    action_map = {pa.patient_id: pa for pa in action.patient_actions}

    icu_correct = 0
    icu_wasted = 0
    total_needing = 0
    icu_used = 0
    doctor_used = 0

    for patient in patients:
        pa = action_map.get(patient.patient_id)
        needs = needs_icu(patient)

        if needs:
            total_needing += 1

        if pa:
            if pa.assign_icu_bed:
                icu_used += 1
                if needs:
                    icu_correct += 1
                else:
                    icu_wasted += 1

            if pa.assign_doctor:
                doctor_used += 1

    # Base efficiency
    efficiency = icu_correct / total_needing if total_needing > 0 else 1.0

    # Misuse penalty
    misuse = icu_wasted / max(1, icu_used)

    # 🔥 STRICT capacity penalty
    capacity_penalty = 0.0
    if icu_used > resources.icu_beds_available:
        overflow = icu_used - resources.icu_beds_available
        capacity_penalty = min(1.0, overflow / max(1, resources.icu_beds_available))

    # 🔥 Doctor overuse penalty
    doctor_penalty = 0.0
    if doctor_used > resources.doctors_available:
        overflow = doctor_used - resources.doctors_available
        doctor_penalty = min(1.0, overflow / max(1, resources.doctors_available))

    # Final resource score
    final_score = efficiency - misuse - capacity_penalty - doctor_penalty
    final_score = max(0.0, final_score)

    return final_score, misuse


# ─────────────────────────────────────────────
# Ignored urgency penalty (ignored_critical)
# ─────────────────────────────────────────────

def _ignored_urgency_fraction(patients: List[Patient], action: Action) -> float:
    urgent = [p for p in patients if p.urgency in (UrgencyLevel.CRITICAL, UrgencyLevel.HIGH)]
    if not urgent:
        return 0.0

    action_ids = {pa.patient_id for pa in action.patient_actions}
    ignored_critical = sum(1 for p in urgent if p.patient_id not in action_ids)

    return ignored_critical / len(urgent)

ignored_critical = _ignored_urgency_fraction


# ─────────────────────────────────────────────
# Triage bonus
# ─────────────────────────────────────────────

def _triage_bonus(
    patients: List[Patient],
    action: Action,
    n_icu_beds: int,
) -> float:

    icu_assigned = {
        pa.patient_id for pa in action.patient_actions if pa.assign_icu_bed
    }

    sorted_by_sev = sorted(patients, key=lambda p: -p.severity)
    optimal = {p.patient_id for p in sorted_by_sev[:n_icu_beds]}

    overlap = len(icu_assigned & optimal)

    return (overlap / max(1, len(optimal))) * 0.10


# ─────────────────────────────────────────────
# EASY
# ─────────────────────────────────────────────

def grade_easy(observation: Observation, action: Action) -> GradeResult:
    treatment = _score_treatments(observation.patients, action)

    resource, misuse = _score_resources(
        observation.patients, action, observation.resources
    )

    priority = _score_priorities(observation.patients, action)

    score = (
        0.60 * treatment +
        0.30 * resource +
        0.10 * priority
    )

    return GradeResult(score, {
        "treatment": treatment,
        "resource": resource,
        "priority": priority,
    })


# ─────────────────────────────────────────────
# MEDIUM
# ─────────────────────────────────────────────

def grade_medium(observation: Observation, action: Action) -> GradeResult:
    treatment = _score_treatments(observation.patients, action)
    priority = _score_priorities(observation.patients, action)

    resource, _ = _score_resources(
        observation.patients, action, observation.resources
    )

    ignored = _ignored_urgency_fraction(observation.patients, action)

    score = (
        0.40 * treatment +
        0.35 * priority +
        0.25 * resource
        - 0.35 * ignored
    )

    return GradeResult(score, {
        "treatment": treatment,
        "priority": priority,
        "resource": resource,
        "ignored_penalty": ignored,
    })


# ─────────────────────────────────────────────
# HARD
# ─────────────────────────────────────────────

def grade_hard(observation: Observation, action: Action) -> GradeResult:
    treatment = _score_treatments(observation.patients, action)
    priority = _score_priorities(observation.patients, action)

    resource, _ = _score_resources(
        observation.patients, action, observation.resources
    )

    ignored = _ignored_urgency_fraction(observation.patients, action)

    triage = _triage_bonus(
        observation.patients,
        action,
        observation.resources.icu_beds_available,
    )

    score = (
        0.35 * treatment +
        0.30 * priority +
        0.35 * resource
        - 0.50 * ignored
        + triage
    )

    return GradeResult(score, {
        "treatment": treatment,
        "priority": priority,
        "resource": resource,
        "ignored_penalty": ignored,
        "triage_bonus": triage,
    })


# ─────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────

GRADER_REGISTRY = {
    "task_easy": grade_easy,
    "task_medium": grade_medium,
    "task_hard": grade_hard,
}