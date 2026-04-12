from __future__ import annotations

from .models import (
    HospitalResources,
    Observation,
    Patient,
    Symptom,
    UrgencyLevel,
)


def get_task_easy() -> Observation:
    return Observation(
        step=0,
        task_id="task_easy",
        task_description="Single critical cardiac patient. Choose correct treatment.",
        patients=[
            Patient(
                patient_id="P001",
                symptoms=[Symptom.CHEST_PAIN, Symptom.SHORTNESS_OF_BREATH],
                severity=0.85,
                urgency=UrgencyLevel.CRITICAL,
                age=62,
                comorbidities=["hypertension", "diabetes"],
                hours_waiting=0.5,
            )
        ],
        resources=HospitalResources(
            icu_beds_available=4,
            icu_beds_total=4,
            general_beds_available=10,
            general_beds_total=10,
            doctors_available=5,
            doctors_total=5,
        ),
        pending_patients=0,
        time_elapsed_hours=0.0,
    )


def get_task_medium() -> Observation:
    return Observation(
        step=0,
        task_id="task_medium",
        task_description="Multi-patient triage with limited ICU and doctors.",
        patients=[
            Patient(
                patient_id="P001",
                symptoms=[Symptom.STROKE_SYMPTOMS],
                severity=0.90,
                urgency=UrgencyLevel.CRITICAL,
                age=71,
                comorbidities=["af"],
                hours_waiting=0.2,
            ),
            Patient(
                patient_id="P002",
                symptoms=[Symptom.FRACTURE],
                severity=0.35,
                urgency=UrgencyLevel.MODERATE,
                age=34,
                comorbidities=[],
                hours_waiting=2.0,
            ),
            Patient(
                patient_id="P003",
                symptoms=[Symptom.HIGH_FEVER, Symptom.NAUSEA],
                severity=0.55,
                urgency=UrgencyLevel.HIGH,
                age=45,
                comorbidities=["immune"],
                hours_waiting=1.0,
            ),
            Patient(
                patient_id="P004",
                symptoms=[Symptom.CHEST_PAIN],
                severity=0.80,
                urgency=UrgencyLevel.CRITICAL,
                age=58,
                comorbidities=["lipid"],
                hours_waiting=0.1,
            ),
            Patient(
                patient_id="P005",
                symptoms=[Symptom.HEADACHE],
                severity=0.20,
                urgency=UrgencyLevel.LOW,
                age=28,
                comorbidities=[],
                hours_waiting=3.5,
            ),
        ],
        resources=HospitalResources(
            icu_beds_available=2,
            icu_beds_total=4,
            general_beds_available=3,
            general_beds_total=8,
            doctors_available=3,
            doctors_total=6,
        ),
        pending_patients=3,
        time_elapsed_hours=1.0,
    )


def get_task_hard() -> Observation:
    return Observation(
        step=0,
        task_id="task_hard",
        task_description="Critical overload with ICU scarcity and dynamic spike.",
        patients=[
            Patient(
                patient_id="P001",
                symptoms=[Symptom.TRAUMA],
                severity=0.95,
                urgency=UrgencyLevel.CRITICAL,
                age=22,
                comorbidities=[],
                hours_waiting=0.05,
            ),
            Patient(
                patient_id="P002",
                symptoms=[Symptom.CHEST_PAIN, Symptom.SHORTNESS_OF_BREATH],
                severity=0.88,
                urgency=UrgencyLevel.CRITICAL,
                age=67,
                comorbidities=["hf"],
                hours_waiting=0.3,
            ),
            Patient(
                patient_id="P003",
                symptoms=[Symptom.STROKE_SYMPTOMS],
                severity=0.82,
                urgency=UrgencyLevel.CRITICAL,
                age=79,
                comorbidities=[],
                hours_waiting=0.1,
            ),
            Patient(
                patient_id="P004",
                symptoms=[Symptom.ABDOMINAL_PAIN],
                severity=0.78,
                urgency=UrgencyLevel.CRITICAL,
                age=41,
                comorbidities=[],
                hours_waiting=0.8,
            ),
            Patient(
                patient_id="P005",
                symptoms=[Symptom.HIGH_FEVER],
                severity=0.65,
                urgency=UrgencyLevel.HIGH,
                age=55,
                comorbidities=[],
                hours_waiting=1.2,
            ),
            Patient(
                patient_id="P006",
                symptoms=[Symptom.FRACTURE],
                severity=0.50,
                urgency=UrgencyLevel.MODERATE,
                age=33,
                comorbidities=[],
                hours_waiting=2.0,
            ),
            Patient(
                patient_id="P007",
                symptoms=[Symptom.MILD_INFECTION],
                severity=0.30,
                urgency=UrgencyLevel.LOW,
                age=48,
                comorbidities=[],
                hours_waiting=4.0,
            ),
            Patient(
                patient_id="P008",
                symptoms=[Symptom.HEADACHE],
                severity=0.15,
                urgency=UrgencyLevel.LOW,
                age=19,
                comorbidities=[],
                hours_waiting=5.0,
            ),
        ],
        resources=HospitalResources(
            icu_beds_available=2,
            icu_beds_total=8,
            general_beds_available=3,
            general_beds_total=10,
            doctors_available=3,
            doctors_total=8,
        ),
        pending_patients=6,
        time_elapsed_hours=2.0,
    )


TASK_REGISTRY = {
    "task_easy": get_task_easy,
    "task_medium": get_task_medium,
    "task_hard": get_task_hard,
}