from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Tuple

from .graders import GRADER_REGISTRY
from .models import (
    Action,
    Observation,
    Patient,
    Reward,
    RewardBreakdown,
    UrgencyLevel,
    Symptom,
)
from .tasks import TASK_REGISTRY
from .utils import (
    compute_wait_penalty,
    needs_icu,
    optimal_priority_order,
    priority_rank_score,
    treatment_score,
)

# ─────────────────────────────────────────────
# Reward weights
# ─────────────────────────────────────────────

W_CORRECT_TREATMENT = 0.40
W_ACCEPTABLE_TREAT = 0.20
W_WRONG_TREATMENT = -0.25
W_OVERTREATMENT = -0.35
W_HARMFUL_TREATMENT = -0.50

W_CRITICAL_BONUS = 0.20
W_IGNORED_CRITICAL = -0.40

W_ICU_CORRECT = 0.15
W_ICU_WASTED = -0.20
W_ICU_MISSED = -0.25

W_PRIORITY_MAX = 0.10

W_WAIT_THRESHOLD = 2.0
W_WAIT_RATE = 0.05


class HospitalEnv:
    def __init__(self, task_id: str, max_steps: Optional[int] = None):
        if task_id not in TASK_REGISTRY:
            raise ValueError(f"Unknown task_id '{task_id}'")

        self.task_id = task_id
        self._max_steps_override = max_steps
        self._current_obs: Optional[Observation] = None
        self._step_count = 0
        self._done = False

    # ─────────────────────────────
    # STATE (FIXED ✅)
    # ─────────────────────────────
    def state(self):
        """Return current environment state"""
        if self._current_obs is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        obs = self._current_obs

        return {
            "step": obs.step,
            "patients": [p.model_dump() for p in obs.patients],
            "resources": obs.resources.model_dump(),
            "pending_patients": obs.pending_patients,
            "time_elapsed_hours": obs.time_elapsed_hours,
            "task_id": obs.task_id,
        }

    # ─────────────────────────────
    # RESET
    # ─────────────────────────────
    def reset(self) -> Observation:
        obs = TASK_REGISTRY[self.task_id]()
        self._current_obs = copy.deepcopy(obs)
        self._step_count = 0
        self._done = False
        return copy.deepcopy(obs)

    # ─────────────────────────────
    # STEP
    # ─────────────────────────────
    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        if self._current_obs is None:
            raise RuntimeError("Call reset() first.")
        if self._done:
            raise RuntimeError("Episode finished.")

        obs = self._current_obs
        grader = GRADER_REGISTRY[self.task_id]

        # Resource validation
        icu_assigned = sum(pa.assign_icu_bed for pa in action.patient_actions)
        gen_assigned = sum(pa.assign_general_bed for pa in action.patient_actions)
        doc_assigned = sum(pa.assign_doctor for pa in action.patient_actions)

        penalty = 0.0

        if icu_assigned > obs.resources.icu_beds_available:
            penalty += 0.3 * (icu_assigned - obs.resources.icu_beds_available)

        if gen_assigned > obs.resources.general_beds_available:
            penalty += 0.2 * (gen_assigned - obs.resources.general_beds_available)

        if doc_assigned > obs.resources.doctors_available:
            penalty += 0.2 * (doc_assigned - obs.resources.doctors_available)

        # Reward
        reward = self._compute_reward(action, obs)
        reward.value = max(-1.0, reward.value - penalty)
        reward.normalized = (reward.value + 1.0) / 2.0

        # Grader
        grade = grader(obs, action)

        # State update
        new_obs = self._apply_action(action, obs)

        self._step_count += 1

        max_steps = self._max_steps_override or _task_max_steps(self.task_id)
        if self._step_count >= max_steps or len(new_obs.patients) == 0:
            self._done = True

        self._current_obs = new_obs

        return copy.deepcopy(new_obs), reward, self._done, {
            "grade": grade.score,
            "step": self._step_count,
        }

    # ─────────────────────────────
    # REWARD
    # ─────────────────────────────
    def _compute_reward(self, action: Action, obs: Observation) -> Reward:
        patients = obs.patients
        resources = obs.resources
        action_map = {pa.patient_id: pa for pa in action.patient_actions}
        optimal_order = optimal_priority_order(patients)

        raw_total = 0.0
        n = len(patients) if patients else 1
        breakdown = RewardBreakdown()

        for patient in patients:
            pa = action_map.get(patient.patient_id)
            is_critical = patient.urgency in (UrgencyLevel.CRITICAL, UrgencyLevel.HIGH)

            if pa is None:
                raw_total += (W_IGNORED_CRITICAL if is_critical else -0.1) / n
                continue

            # Treatment
            t_raw = treatment_score(patient, pa.treatment)

            if t_raw >= 1:
                raw_total += W_CORRECT_TREATMENT / n
            elif t_raw >= 0.5:
                raw_total += W_ACCEPTABLE_TREAT / n
            elif t_raw <= -1:
                raw_total += W_HARMFUL_TREATMENT / n
            elif t_raw <= -0.5:
                raw_total += W_OVERTREATMENT / n
            else:
                raw_total += W_WRONG_TREATMENT / n

            # ICU
            if pa.assign_icu_bed:
                if needs_icu(patient):
                    raw_total += W_ICU_CORRECT / n
                else:
                    raw_total += W_ICU_WASTED / n
            else:
                if needs_icu(patient) and resources.icu_beds_available > 0:
                    raw_total += W_ICU_MISSED / n

            # Priority
            p_score = priority_rank_score(
                patient.patient_id, pa.priority_rank, optimal_order
            )
            raw_total += ((p_score * 2 - 1) * W_PRIORITY_MAX) / n

            # Wait penalty
            raw_total += compute_wait_penalty(
                patient.hours_waiting, W_WAIT_THRESHOLD, W_WAIT_RATE
            ) / n

        clamped = max(-1.0, min(1.0, raw_total))

        return Reward(
            value=round(clamped, 6),
            normalized=round((clamped + 1) / 2, 6),
            breakdown=breakdown,
        )

    # ─────────────────────────────
    # STATE TRANSITION
    # ─────────────────────────────
    def _apply_action(self, action: Action, obs: Observation) -> Observation:
        new_obs = copy.deepcopy(obs)

        action_map = {pa.patient_id: pa for pa in action.patient_actions}

        remaining = []
        icu_used = 0
        gen_used = 0
        doc_used = 0

        for patient in new_obs.patients:
            pa = action_map.get(patient.patient_id)

            if pa is None:
                remaining.append(patient)
                continue

            if pa.assign_icu_bed:
                icu_used += 1
            if pa.assign_general_bed:
                gen_used += 1
            if pa.assign_doctor:
                doc_used += 1

            # remove if treated properly
            if treatment_score(patient, pa.treatment) >= 0.5 and pa.assign_doctor:
                continue
            else:
                remaining.append(patient)

        # Update resources
        new_obs.resources.icu_beds_available = max(
            0, new_obs.resources.icu_beds_available - icu_used
        )
        new_obs.resources.general_beds_available = max(
            0, new_obs.resources.general_beds_available - gen_used
        )
        new_obs.resources.doctors_available = max(
            0, new_obs.resources.doctors_available - doc_used
        )

        # Aging patients
        for p in remaining:
            p.hours_waiting += 0.5
            if p.urgency in (UrgencyLevel.CRITICAL, UrgencyLevel.HIGH):
                p.severity = min(1.0, p.severity + 0.05)

        # Hard task dynamic event
        if self.task_id == "task_hard" and self._step_count == 1:
            remaining.append(_emergency_spike_patient())

        new_obs.patients = remaining
        new_obs.step = self._step_count + 1
        new_obs.time_elapsed_hours += 0.5

        return new_obs


# ─────────────────────────────
# HELPERS
# ─────────────────────────────

def _task_max_steps(task_id: str) -> int:
    return {"task_easy": 1, "task_medium": 1, "task_hard": 3}.get(task_id, 1)


def _emergency_spike_patient() -> Patient:
    return Patient(
        patient_id="P009",
        symptoms=[Symptom.TRAUMA],
        severity=0.91,
        urgency=UrgencyLevel.CRITICAL,
        age=35,
        comorbidities=[],
        hours_waiting=0.0,
    )