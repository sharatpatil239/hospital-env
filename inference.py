#!/usr/bin/env python3

"""
Hospital OpenEnv - LLM Inference

Runs the three hospital decision-making tasks using an OpenRouter LLM.

Usage:
    1. Start server.py in Terminal 1
    2. Start inference.py in Terminal 2
    3. Paste your OpenRouter API key when prompted

The API key is entered using normal input(), so Ctrl+V / Ctrl+Shift+V
works normally in the VS Code terminal.
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
    TreatmentType,
)
from environment.utils import needs_icu


# ============================================================
# CONFIGURATION
# ============================================================

API_BASE_URL = os.environ.get(
    "API_BASE_URL",
    "https://openrouter.ai/api/v1",
)

MODEL_NAME = os.environ.get(
    "MODEL_NAME",
    "openai/gpt-4o-mini",
)

_client: Optional[OpenAI] = None


# ============================================================
# API KEY
# ============================================================

def get_api_key() -> str:
    """
    Get OpenRouter API key.

    Priority:
        1. OPENROUTER_API_KEY environment variable
        2. OPENAI_API_KEY environment variable
        3. Manual terminal input

    Manual input uses input(), NOT getpass(), so the key can
    be pasted normally in VS Code.
    """

    api_key = (
        os.getenv("OPENROUTER_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or ""
    ).strip()

    if not api_key:
        print()
        print("=" * 60)
        print("OPENROUTER API KEY REQUIRED")
        print("=" * 60)
        print("Paste your OpenRouter API key below.")
        print("Your input will be visible in the terminal.")
        print()

        try:
            api_key = input("Enter your OpenRouter API key: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[ERROR] API key input cancelled.")
            sys.exit(1)

    if not api_key:
        print("[ERROR] OpenRouter API key is missing or empty.")
        sys.exit(1)

    return api_key


# ============================================================
# OPENROUTER CLIENT
# ============================================================

def get_client() -> OpenAI:
    """Create the OpenRouter OpenAI-compatible client."""

    global _client

    if _client is None:

        api_key = get_api_key()

        print()
        print("[INFO] Initializing OpenRouter client...")
        print(f"[INFO] API base URL: {API_BASE_URL}")
        print(f"[INFO] Model: {MODEL_NAME}")
        print()

        _client = OpenAI(
            base_url=API_BASE_URL,
            api_key=api_key,
            default_headers={
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "Hospital OpenEnv",
            },
        )

    return _client


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a senior hospital attending physician and triage coordinator.

You will receive a JSON object describing the current state of a
hospital emergency department.

You must respond with ONLY one valid JSON object.

Do NOT use markdown.
Do NOT use ```json.
Do NOT provide explanations outside the JSON object.

Required format:

{
  "patient_actions": [
    {
      "patient_id": "P001",
      "treatment": "cardiac_intervention",
      "assign_icu_bed": true,
      "assign_general_bed": false,
      "assign_doctor": true,
      "priority_rank": 1
    }
  ],
  "reasoning": "Brief justification"
}

VALID TREATMENTS:

emergency_surgery
cardiac_intervention
icu_monitoring
antibiotics
pain_management
respiratory_support
neurological_eval
orthopedic_care
observation
discharge

RULES:

1. Include EVERY patient.

2. priority_rank must be unique.

3. priority_rank starts at 1.

4. Rank patients by:
   CRITICAL > HIGH > MODERATE > LOW

5. For equal urgency, higher severity goes first.

6. ICU should be assigned only when clinically appropriate.

7. ICU candidates:
   - chest_pain
   - shortness_of_breath
   - stroke_symptoms
   - trauma with severity >= 0.7
   - any patient with severity >= 0.85

8. Never discharge a patient with severity > 0.6.

9. Match treatment to symptoms.

SYMPTOM MAPPING:

chest_pain
-> cardiac_intervention

shortness_of_breath
-> respiratory_support

stroke_symptoms
-> neurological_eval

trauma
-> emergency_surgery

abdominal_pain
-> pain_management or emergency_surgery

high_fever
-> antibiotics

fracture
-> orthopedic_care

mild_infection
-> antibiotics

headache
-> observation or pain_management

nausea
-> observation or pain_management
"""


# ============================================================
# OBSERVATION
# ============================================================

def build_observation_prompt(obs: Observation) -> str:
    """Convert environment observation into JSON for the LLM."""

    data = {
        "task": obs.task_id,
        "description": obs.task_description,
        "step": obs.step,
        "time_elapsed_hours": obs.time_elapsed_hours,
        "pending_patients": obs.pending_patients,

        "resources": {
            "icu_beds_available": obs.resources.icu_beds_available,
            "icu_beds_total": obs.resources.icu_beds_total,
            "general_beds_available": obs.resources.general_beds_available,
            "general_beds_total": obs.resources.general_beds_total,
            "doctors_available": obs.resources.doctors_available,
            "doctors_total": obs.resources.doctors_total,
        },

        "patients": [
            {
                "patient_id": p.patient_id,
                "symptoms": [s.value for s in p.symptoms],
                "severity": p.severity,
                "urgency": p.urgency.value,
                "age": p.age,
                "comorbidities": p.comorbidities,
                "hours_waiting": p.hours_waiting,
            }
            for p in obs.patients
        ],
    }

    return json.dumps(data, indent=2)


# ============================================================
# LLM CALL
# ============================================================

def call_llm(user_content: str) -> str:
    """Send observation to OpenRouter."""

    try:

        response = get_client().chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
            temperature=0.0,
            max_tokens=1500,
        )

        content = response.choices[0].message.content

        if not content:
            print("[ERROR] LLM returned an empty response.", file=sys.stderr)
            return ""

        return content.strip()

    except Exception as exc:

        print(
            f"[ERROR] OpenRouter API call failed: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )

        return ""


# ============================================================
# ACTION PARSING
# ============================================================

def parse_action(raw_response: str, obs: Observation) -> Action:
    """Convert LLM JSON response into Action."""

    if not raw_response:
        return _fallback_action(obs)

    cleaned = raw_response.strip()

    # Remove markdown fences if model ignores the instruction.
    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
    )

    try:

        data = json.loads(cleaned)

        action = Action(**data)

        return _fill_missing_patients(action, obs)

    except Exception as exc:

        print(
            f"[WARN] Could not parse LLM action: {exc}",
            file=sys.stderr,
        )

        return _fallback_action(obs)


# ============================================================
# FILL MISSING PATIENTS
# ============================================================

def _fill_missing_patients(
    action: Action,
    obs: Observation,
) -> Action:

    patient_ids = {
        pa.patient_id
        for pa in action.patient_actions
    }

    missing = [
        patient
        for patient in obs.patients
        if patient.patient_id not in patient_ids
    ]

    if not missing:
        return action

    existing = list(action.patient_actions)

    used_ranks = {
        pa.priority_rank
        for pa in existing
    }

    next_rank = max(used_ranks, default=0) + 1

    for patient in missing:

        treatment = _default_treatment(patient)

        icu_used = sum(
            1
            for pa in existing
            if pa.assign_icu_bed
        )

        assign_icu = (
            needs_icu(patient)
            and icu_used < obs.resources.icu_beds_available
        )

        existing.append(
            PatientAction(
                patient_id=patient.patient_id,
                treatment=treatment,
                assign_icu_bed=assign_icu,
                assign_general_bed=not assign_icu,
                assign_doctor=True,
                priority_rank=next_rank,
            )
        )

        next_rank += 1

    return Action(
        patient_actions=existing,
        reasoning=action.reasoning,
    )


# ============================================================
# FALLBACK
# ============================================================

def _fallback_action(obs: Observation) -> Action:
    """
    Deterministic fallback if OpenRouter fails.

    This allows the environment to continue instead of crashing.
    """

    icu_budget = obs.resources.icu_beds_available
    icu_assigned = 0

    actions = []

    sorted_patients = sorted(
        obs.patients,
        key=lambda p: (
            -p.severity,
            -p.hours_waiting,
        ),
    )

    for rank, patient in enumerate(
        sorted_patients,
        start=1,
    ):

        treatment = _default_treatment(patient)

        assign_icu = (
            needs_icu(patient)
            and icu_assigned < icu_budget
        )

        if assign_icu:
            icu_assigned += 1

        actions.append(
            PatientAction(
                patient_id=patient.patient_id,
                treatment=treatment,
                assign_icu_bed=assign_icu,
                assign_general_bed=not assign_icu,
                assign_doctor=True,
                priority_rank=rank,
            )
        )

    return Action(
        patient_actions=actions,
        reasoning="fallback_heuristic",
    )


# ============================================================
# DEFAULT TREATMENT
# ============================================================

def _default_treatment(patient) -> TreatmentType:

    defaults = {
        "chest_pain":
            TreatmentType.CARDIAC_INTERVENTION,

        "shortness_of_breath":
            TreatmentType.RESPIRATORY_SUPPORT,

        "high_fever":
            TreatmentType.ANTIBIOTICS,

        "trauma":
            TreatmentType.EMERGENCY_SURGERY,

        "stroke_symptoms":
            TreatmentType.NEUROLOGICAL_EVAL,

        "abdominal_pain":
            TreatmentType.PAIN_MANAGEMENT,

        "fracture":
            TreatmentType.ORTHOPEDIC_CARE,

        "mild_infection":
            TreatmentType.ANTIBIOTICS,

        "headache":
            TreatmentType.OBSERVATION,

        "nausea":
            TreatmentType.OBSERVATION,
    }

    if not patient.symptoms:
        return TreatmentType.OBSERVATION

    symptom = patient.symptoms[0].value

    return defaults.get(
        symptom,
        TreatmentType.OBSERVATION,
    )


# ============================================================
# LOGGING
# ============================================================

def log_start(task_name: str) -> None:

    print(
        f"[START] task: {task_name}",
        flush=True,
    )


def log_step(
    observation: Observation,
    action: Action,
    reward: float,
) -> None:

    obs_json = json.dumps(
        {
            "step": observation.step,
            "task_id": observation.task_id,
            "n_patients": len(observation.patients),
            "icu_beds_available":
                observation.resources.icu_beds_available,
        },
        separators=(",", ":"),
    )

    action_json = json.dumps(
        [
            pa.model_dump()
            for pa in action.patient_actions
        ],
        separators=(",", ":"),
    )

    print(
        f"[STEP] observation: {obs_json} "
        f"action: {action_json} "
        f"reward: {reward:.4f}",
        flush=True,
    )


def log_end(score: float) -> None:

    print(
        f"[END] score: {score:.4f}",
        flush=True,
    )


# ============================================================
# RUN TASK
# ============================================================

def run_task(task_id: str) -> Dict[str, Any]:

    log_start(task_id)

    env = HospitalEnv(task_id=task_id)

    obs = env.reset()

    total_reward = 0.0
    final_score = 0.0
    step_count = 0

    try:

        done = False

        while not done:

            prompt = build_observation_prompt(obs)

            raw_response = call_llm(prompt)

            action = parse_action(
                raw_response,
                obs,
            )

            next_obs, reward, done, info = env.step(action)

            total_reward += reward.value

            final_score = float(
                info.get("grade", 0.0)
            )

            step_count += 1

            log_step(
                obs,
                action,
                reward.value,
            )

            obs = next_obs

    except Exception as exc:

        print(
            f"[ERROR] task={task_id} error={exc}",
            file=sys.stderr,
        )

        traceback.print_exc(
            file=sys.stderr
        )

        try:

            log_step(
                obs,
                Action(
                    patient_actions=[],
                    reasoning="error",
                ),
                0.0,
            )

        except Exception:
            pass

    log_end(final_score)

    return {
        "task_id": task_id,
        "score": final_score,
        "total_reward": total_reward,
        "steps": step_count,
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 60)
    print("HOSPITAL OPENENV - LLM INFERENCE")
    print("=" * 60)
    print()

    # Validate API connection/client before benchmark.
    get_client()

    task_ids = [
        "task_easy",
        "task_medium",
        "task_hard",
    ]

    results: List[Dict[str, Any]] = []

    for task_id in task_ids:

        result = run_task(task_id)

        results.append(result)

        print("", flush=True)

    # Summary goes to stderr so benchmark stdout remains clean.

    avg_score = (
        sum(r["score"] for r in results)
        / len(results)
    )

    print(
        "\n# Summary: "
        f"avg_score={avg_score:.4f} | "
        + " | ".join(
            f"{r['task_id']}={r['score']:.4f}"
            for r in results
        ),
        file=sys.stderr,
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()