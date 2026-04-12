#!/usr/bin/env python3
"""
inference.py — Baseline inference script for the Hospital Decision-Making OpenEnv.

Reads configuration from environment variables:
    API_BASE_URL    : LLM API base URL (default: https://api.openai.com/v1)
    MODEL_NAME      : Model identifier  (default: gpt-4o)
    OPENAI_API_KEY  : API key (also checked as HF_TOKEN for backwards compatibility)

Runs ALL THREE tasks (easy, medium, hard) sequentially.
Each task uses the LLM to decide actions step-by-step.

STRICTLY ENFORCED log format:
    [START] task: <task_name>
    [STEP] observation: <json> action: <json> reward: <float>
    [END] score: <float>

Any deviation from this format is NOT allowed.

Runtime guarantee: < 20 minutes total (all 3 tasks).
"""

from __future__ import annotations

import json
import os
import re
import sys
import traceback
from typing import Any, Dict, List, Optional

from openai import OpenAI

from environment import (
    Action,
    HospitalEnv,
    Observation,
    PatientAction,
    GRADER_REGISTRY,
    TASK_REGISTRY,
    TreatmentType,
    UrgencyLevel,
)
from environment.utils import needs_icu


API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME:   str = os.environ.get("MODEL_NAME",   "gpt-4o")

OPENAI_API_KEY: str = (
    os.environ.get("OPENAI_API_KEY")
    or os.environ.get("HF_TOKEN")
    or "none"
)

_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    """Lazily initialise the OpenAI client."""
    global _client
    if _client is None:
        _client = OpenAI(base_url=API_BASE_URL, api_key=OPENAI_API_KEY)
    return _client




SYSTEM_PROMPT = """You are a senior hospital attending physician and triage coordinator.
You will receive a JSON object describing the current state of the emergency department,
including patients and available resources.

You must respond with a single valid JSON object representing your clinical decisions.
No markdown fences, no explanation, no preamble — raw JSON only.

RESPONSE FORMAT:
{
  "patient_actions": [
    {
      "patient_id": "P001",
      "treatment": "<treatment_type>",
      "assign_icu_bed": true,
      "assign_general_bed": false,
      "assign_doctor": true,
      "priority_rank": 1
    }
  ],
  "reasoning": "Brief justification"
}

VALID TREATMENT TYPES (choose exactly one per patient):
  emergency_surgery    — major trauma, internal bleeding, acute abdomen
  cardiac_intervention — chest pain, heart attack, cardiac emergency
  icu_monitoring       — critical patients requiring continuous monitoring
  antibiotics          — infection (fever, sepsis, mild infection)
  pain_management      — fractures, abdominal pain, headache
  respiratory_support  — shortness of breath, respiratory distress
  neurological_eval    — stroke symptoms, neurological emergencies
  orthopedic_care      — fractures, musculoskeletal injuries
  observation          — low-severity, monitoring only
  discharge            — stable, non-urgent, can safely leave

CLINICAL RULES (strictly follow):
1. Include EVERY patient in patient_actions — no patient may be omitted.
2. priority_rank must be unique integers starting at 1 (rank 1 = treat first).
3. Prioritize: CRITICAL > HIGH > MODERATE > LOW urgency; break ties by severity (higher first).
4. assign_icu_bed = true ONLY for: chest_pain, shortness_of_breath, stroke_symptoms,
   trauma patients with severity ≥ 0.7, OR any patient with severity ≥ 0.85.
5. assign_icu_bed = false for all other patients (do NOT waste ICU beds).
6. Never discharge a patient with severity > 0.6.
7. Match treatment to primary symptom (see valid types above).

SYMPTOM → TREATMENT GUIDE:
  chest_pain            → cardiac_intervention  (ICU bed required)
  shortness_of_breath   → respiratory_support   (ICU bed required)
  stroke_symptoms       → neurological_eval     (ICU bed required)
  trauma                → emergency_surgery     (ICU bed if severity ≥ 0.7)
  abdominal_pain        → emergency_surgery or pain_management
  high_fever            → antibiotics
  fracture              → orthopedic_care
  mild_infection        → antibiotics
  headache / nausea     → pain_management or observation
"""


def build_observation_prompt(obs: Observation) -> str:
    """Serialize observation to a compact JSON string for the LLM."""
    data = {
        "task":                 obs.task_id,
        "description":          obs.task_description,
        "step":                 obs.step,
        "time_elapsed_hours":   obs.time_elapsed_hours,
        "pending_patients":     obs.pending_patients,
        "resources": {
            "icu_beds_available":      obs.resources.icu_beds_available,
            "icu_beds_total":          obs.resources.icu_beds_total,
            "general_beds_available":  obs.resources.general_beds_available,
            "general_beds_total":      obs.resources.general_beds_total,
            "doctors_available":       obs.resources.doctors_available,
            "doctors_total":           obs.resources.doctors_total,
        },
        "patients": [
            {
                "patient_id":    p.patient_id,
                "symptoms":      [s.value for s in p.symptoms],
                "severity":      p.severity,
                "urgency":       p.urgency.value,
                "age":           p.age,
                "comorbidities": p.comorbidities,
                "hours_waiting": p.hours_waiting,
            }
            for p in obs.patients
        ],
    }
    return json.dumps(data, indent=2)



def call_llm(user_content: str) -> str:
    """
    Call the language model with the system prompt and observation.
    Returns the raw text response.
    Temperature = 0 for reproducible, deterministic output.
    """
    response = get_client().chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        temperature=0.0,
        max_tokens=1500,
    )
    return response.choices[0].message.content.strip()



def parse_action(raw_response: str, obs: Observation) -> Action:
    """
    Parse the LLM's JSON response into a validated Action object.

    Falls back to a deterministic heuristic action if parsing fails,
    ensuring the episode always progresses (no crash on bad LLM output).
    """
    # Strip markdown fences if present
    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$", "", raw_response.strip(), flags=re.MULTILINE
    ).strip()

    try:
        data   = json.loads(cleaned)
        action = Action(**data)
        # Verify all patients have an action — fill gaps if needed
        action = _fill_missing_patients(action, obs)
        return action
    except Exception as exc:
        # Log parse error to stderr (does not pollute stdout log format)
        print(f"[WARN] Action parse failed: {exc}. Using fallback.", file=sys.stderr)
        return _fallback_action(obs)


def _fill_missing_patients(action: Action, obs: Observation) -> Action:
    """
    Ensure every patient in obs has a corresponding PatientAction.
    Any patient missing from the action gets a safe default entry appended.
    """
    covered_ids = {pa.patient_id for pa in action.patient_actions}
    missing     = [p for p in obs.patients if p.patient_id not in covered_ids]

    if not missing:
        return action

    existing_ranks = {pa.priority_rank for pa in action.patient_actions}
    next_rank      = max(existing_ranks, default=0) + 1
    extra_actions  = list(action.patient_actions)

    for patient in missing:
        treatment  = _default_treatment(patient)
        assign_icu = needs_icu(patient) and (
            sum(1 for pa in extra_actions if pa.assign_icu_bed)
            < obs.resources.icu_beds_available
        )
        extra_actions.append(PatientAction(
            patient_id=patient.patient_id,
            treatment=treatment,
            assign_icu_bed=assign_icu,
            assign_general_bed=not assign_icu,
            assign_doctor=True,
            priority_rank=next_rank,
        ))
        next_rank += 1

    return Action(patient_actions=extra_actions, reasoning=action.reasoning)


def _fallback_action(obs: Observation) -> Action:
    """
    Deterministic fallback: assign best-guess treatments in severity order.
    Used only when the LLM response is completely unparseable.
    """
    icu_budget    = obs.resources.icu_beds_available
    icu_assigned  = 0
    patient_actions = []

    sorted_patients = sorted(obs.patients, key=lambda p: -p.severity)

    for rank, patient in enumerate(sorted_patients, start=1):
        treatment   = _default_treatment(patient)
        assign_icu  = needs_icu(patient) and icu_assigned < icu_budget
        if assign_icu:
            icu_assigned += 1

        patient_actions.append(PatientAction(
            patient_id=patient.patient_id,
            treatment=treatment,
            assign_icu_bed=assign_icu,
            assign_general_bed=not assign_icu,
            assign_doctor=True,
            priority_rank=rank,
        ))

    return Action(patient_actions=patient_actions, reasoning="fallback_heuristic")


def _default_treatment(patient) -> TreatmentType:
    """
    Rule-based treatment selection for fallback/gap-filling.
    Maps primary symptom to most appropriate default treatment.
    """
    defaults = {
        "chest_pain":          TreatmentType.CARDIAC_INTERVENTION,
        "shortness_of_breath": TreatmentType.RESPIRATORY_SUPPORT,
        "high_fever":          TreatmentType.ANTIBIOTICS,
        "trauma":              TreatmentType.EMERGENCY_SURGERY,
        "stroke_symptoms":     TreatmentType.NEUROLOGICAL_EVAL,
        "abdominal_pain":      TreatmentType.PAIN_MANAGEMENT,
        "fracture":            TreatmentType.ORTHOPEDIC_CARE,
        "mild_infection":      TreatmentType.ANTIBIOTICS,
        "headache":            TreatmentType.OBSERVATION,
        "nausea":              TreatmentType.OBSERVATION,
    }
    primary_symptom = patient.symptoms[0].value
    return defaults.get(primary_symptom, TreatmentType.OBSERVATION)



def log_start(task_name: str) -> None:
    """Print [START] line. STRICTLY: 'task: <name>'"""
    print(f"[START] task: {task_name}", flush=True)


def log_step(
    observation: Observation,
    action: Action,
    reward: float,
) -> None:
    """
    Print [STEP] line.
    STRICTLY: 'observation: <json> action: <json> reward: <float>'
    """
    obs_json    = json.dumps(
        {
            "step":    observation.step,
            "task_id": observation.task_id,
            "n_patients": len(observation.patients),
            "icu_beds_available": observation.resources.icu_beds_available,
        },
        separators=(",", ":"),
    )
    action_json = json.dumps(
        [pa.model_dump() for pa in action.patient_actions],
        separators=(",", ":"),
    )
    print(
        f"[STEP] observation: {obs_json} action: {action_json} reward: {reward:.4f}",
        flush=True,
    )


def log_end(score: float) -> None:
    """Print [END] line. STRICTLY: 'score: <float>'"""
    print(f"[END] score: {score:.4f}", flush=True)



def run_task(task_id: str) -> Dict[str, Any]:
    """
    Run a complete episode for the given task using the LLM as the agent.

    Logs strictly formatted [START], [STEP]×N, [END] to stdout.
    Errors are sent to stderr to preserve stdout format.

    Returns a dict with task results.
    """
    log_start(task_id)

    env          = HospitalEnv(task_id=task_id)
    obs          = env.reset()
    total_reward = 0.0
    final_score  = 0.0
    step_count   = 0

    try:
        done = False
        while not done:
            # ── Build prompt and call LLM ────────────────────────────────
            prompt     = build_observation_prompt(obs)
            raw_resp   = call_llm(prompt)
            action     = parse_action(raw_resp, obs)

            # ── Step the environment ─────────────────────────────────────
            next_obs, reward, done, info = env.step(action)

            total_reward += reward.value
            final_score   = float(info.get("grade", 0.0))
            step_count   += 1

            # ── Emit STEP log ────────────────────────────────────────────
            log_step(obs, action, reward.value)

            obs = next_obs

    except Exception as exc:
        print(f"[ERROR] task={task_id} error={exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        # Emit a final STEP with zeroed reward so format is never broken
        try:
            log_step(
                obs,
                Action(patient_actions=[], reasoning="error"),
                0.0,
            )
        except Exception:
            pass

    log_end(final_score)

    return {
        "task_id":      task_id,
        "score":        final_score,
        "total_reward": total_reward,
        "steps":        step_count,
    }



def main() -> None:
    """Run all three tasks and summarize results to stderr."""
    task_ids = ["task_easy", "task_medium", "task_hard"]
    results: List[Dict[str, Any]] = []

    for task_id in task_ids:
        result = run_task(task_id)
        results.append(result)
        # Blank line between tasks (stdout only)
        print("", flush=True)

    # Summary — stderr only (keeps stdout format clean)
    avg_score = sum(r["score"] for r in results) / len(results)
    print(
        f"\n# Summary: avg_score={avg_score:.4f} | "
        + " | ".join(f"{r['task_id']}={r['score']:.4f}" for r in results),
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
