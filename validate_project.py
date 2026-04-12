#!/usr/bin/env python3
"""
validate_project.py — Static + structural validation of the hospital-openenv project.

Checks performed (no pydantic/external deps required):
  1. All required files exist
  2. openenv.yaml structure and values
  3. inference.py: correct env vars, correct log format strings, no broken URLs
  4. Dockerfile: non-root user, port 7860, healthcheck
  5. Python files: syntax check via ast.parse
  6. Reward normalization formula present
  7. All 3 tasks defined in tasks.py
  8. All 3 graders defined in graders.py
  9. server.py: all required endpoints
 10. models.py: Reward, Observation, Action classes present

Run from the project root:
    python validate_project.py
"""

from __future__ import annotations

import ast
import os
import re
import sys
import yaml
from pathlib import Path
from typing import List, Tuple


ROOT = Path(__file__).parent
PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "

errors: List[str] = []
warnings: List[str] = []
checks_run = 0


def check(condition: bool, name: str, detail: str = "") -> bool:
    global checks_run
    checks_run += 1
    if condition:
        print(f"  {PASS}  {name}")
        return True
    else:
        msg = f"{name}" + (f": {detail}" if detail else "")
        errors.append(msg)
        print(f"  {FAIL}  {msg}")
        return False


def warn(condition: bool, name: str, detail: str = "") -> bool:
    global checks_run
    checks_run += 1
    if condition:
        print(f"  {PASS}  {name}")
        return True
    else:
        msg = f"{name}" + (f": {detail}" if detail else "")
        warnings.append(msg)
        print(f"  {WARN} {msg}")
        return True  # warnings don't fail


def read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


# ─────────────────────────────────────────────
# 1. Required files
# ─────────────────────────────────────────────

print("\n=== 1. Required Files ===")
required_files = [
    "environment/__init__.py",
    "environment/env.py",
    "environment/models.py",
    "environment/tasks.py",
    "environment/graders.py",
    "environment/utils.py",
    "inference.py",
    "server.py",
    "openenv.yaml",
    "Dockerfile",
    "requirements.txt",
    "README.md",
]
for f in required_files:
    check((ROOT / f).exists(), f"File exists: {f}")



print("\n=== 2. Python Syntax ===")
py_files = [
    "environment/__init__.py",
    "environment/env.py",
    "environment/models.py",
    "environment/tasks.py",
    "environment/graders.py",
    "environment/utils.py",
    "inference.py",
    "server.py",
]
for f in py_files:
    path = ROOT / f
    if not path.exists():
        check(False, f"Syntax OK: {f}", "file not found")
        continue
    try:
        ast.parse(path.read_text())
        check(True, f"Syntax OK: {f}")
    except SyntaxError as e:
        check(False, f"Syntax OK: {f}", str(e))


# ─────────────────────────────────────────────
# 3. openenv.yaml validation
# ─────────────────────────────────────────────

print("\n=== 3. openenv.yaml ===")
yaml_path = ROOT / "openenv.yaml"
yaml_content = read(yaml_path)
try:
    cfg = yaml.safe_load(yaml_content) or {}
except Exception as e:
    cfg = {}
    check(False, "YAML parses", str(e))

check(bool(cfg.get("name")),        "name field present")
check(bool(cfg.get("version")),     "version field present")
check(bool(cfg.get("description")), "description field present")

tasks_yaml = cfg.get("tasks", [])
check(len(tasks_yaml) == 3, "Exactly 3 tasks defined", f"found {len(tasks_yaml)}")

task_ids_yaml = [t.get("id") for t in tasks_yaml]
check("task_easy"   in task_ids_yaml, "task_easy defined in YAML")
check("task_medium" in task_ids_yaml, "task_medium defined in YAML")
check("task_hard"   in task_ids_yaml, "task_hard defined in YAML")

difficulties = [t.get("difficulty") for t in tasks_yaml]
check("easy"   in difficulties, "Easy difficulty present")
check("medium" in difficulties, "Medium difficulty present")
check("hard"   in difficulties, "Hard difficulty present")

reward_cfg = cfg.get("reward", {})
r_range = reward_cfg.get("range", [])
check(
    r_range == [-1.0, 1.0] or r_range == [-1, 1],
    "reward.range is [-1.0, +1.0]",
    f"got {r_range}",
)
norm_range = reward_cfg.get("normalized_range", [])
check(
    norm_range == [0.0, 1.0] or norm_range == [0, 1],
    "reward.normalized_range is [0.0, 1.0]",
    f"got {norm_range}",
)

eval_cfg = cfg.get("evaluation", {})
check(eval_cfg.get("deterministic") is True,   "evaluation.deterministic = true")
check(eval_cfg.get("reproducible") is True,    "evaluation.reproducible = true")
check(eval_cfg.get("partial_credit") is True,  "evaluation.partial_credit = true")

inf_cfg = cfg.get("inference", {})
env_vars = inf_cfg.get("env_vars", [])
check("OPENAI_API_KEY" in env_vars, "OPENAI_API_KEY in inference.env_vars")
check("API_BASE_URL"   in env_vars, "API_BASE_URL in inference.env_vars")
check("MODEL_NAME"     in env_vars, "MODEL_NAME in inference.env_vars")


# ─────────────────────────────────────────────
# 4. inference.py validation
# ─────────────────────────────────────────────

print("\n=== 4. inference.py ===")
inf_text = read(ROOT / "inference.py")

check("OPENAI_API_KEY" in inf_text,  "Reads OPENAI_API_KEY env var")
check("HF_TOKEN"       in inf_text,  "Reads HF_TOKEN fallback (backwards compat)")
check("API_BASE_URL"   in inf_text,  "Reads API_BASE_URL env var")
check("MODEL_NAME"     in inf_text,  "Reads MODEL_NAME env var")
check("from openai import OpenAI" in inf_text, "Uses OpenAI client")
check('temperature=0'  in inf_text,  "temperature=0 (deterministic)")

# Log format checks - STRICT
check('[START] task:' in inf_text,       "[START] uses 'task:' (colon, not equals)")
check('[STEP] observation:' in inf_text, "[STEP] has 'observation:' field")
check('[END] score:' in inf_text,        "[END] uses 'score:' (colon, not equals)")

# Ensure no broken markdown URLs in config
broken_url_pattern = r'\[[\w.]+\]\(https?://'
broken = re.search(broken_url_pattern, inf_text)
check(not broken, "No broken markdown URLs in inference.py",
      f"Found: {broken.group()}" if broken else "")

# Check env.reset() and env.step() are called
check("env.reset()"  in inf_text, "Calls env.reset()")
check("env.step("    in inf_text, "Calls env.step()")

# Check all 3 tasks are run
check('"task_easy"'   in inf_text, "Runs task_easy")
check('"task_medium"' in inf_text, "Runs task_medium")
check('"task_hard"'   in inf_text, "Runs task_hard")

# Fallback action
check("_fallback_action" in inf_text, "Has fallback action for parse failures")


# ─────────────────────────────────────────────
# 5. models.py validation
# ─────────────────────────────────────────────

print("\n=== 5. models.py ===")
models_text = read(ROOT / "environment/models.py")

check("class Observation(" in models_text, "Observation model defined")
check("class Action("      in models_text, "Action model defined")
check("class Reward("      in models_text, "Reward model defined")
check("class Patient("     in models_text, "Patient model defined")
check("class PatientAction(" in models_text, "PatientAction model defined")
check("class RewardBreakdown(" in models_text, "RewardBreakdown model defined")

# Reward must have value (raw) and normalized fields
check("value:"      in models_text or "value: float" in models_text, "Reward.value field")
check("normalized:" in models_text or "normalized: float" in models_text, "Reward.normalized field")

# Normalization formula
check("(value + 1" in models_text or "(self.value + 1" in models_text,
      "Normalization formula (value+1)/2 present")

# Enum definitions
check("class UrgencyLevel"   in models_text, "UrgencyLevel enum")
check("class TreatmentType"  in models_text, "TreatmentType enum")
check("class Symptom"        in models_text, "Symptom enum")

# Range validation for severity
check("ge=0.0" in models_text and "le=1.0" in models_text, "severity bounded [0,1]")


# ─────────────────────────────────────────────
# 6. env.py validation
# ─────────────────────────────────────────────

print("\n=== 6. env.py ===")
env_text = read(ROOT / "environment/env.py")

check("def reset("  in env_text, "reset() method defined")
check("def step("   in env_text, "step() method defined")
check("def state("  in env_text, "state() method defined")

# step() must return 4-tuple
check("return copy.deepcopy(new_obs), reward, self._done, info" in env_text or
      "return copy.deepcopy" in env_text,
      "step() returns (obs, reward, done, info)")

# Reward clamping
check("max(-1.0, min(1.0," in env_text, "Reward clamped to [-1, +1]")

# Normalization in reward
check("(clamped + 1.0) / 2.0" in env_text or
      "(value + 1" in env_text or
      "normalized" in env_text,
      "Normalized reward computed")

# Time simulation
check("hours_waiting" in env_text, "hours_waiting updated (time simulation)")
check("severity" in env_text,      "severity updated over time")

# Dynamic events for hard task
check("task_hard" in env_text and "P009" in env_text,
      "Dynamic event (P009 spike) in hard task")

# Grader called in step
check("grader(" in env_text, "Grader called in step()")


# ─────────────────────────────────────────────
# 7. tasks.py validation
# ─────────────────────────────────────────────

print("\n=== 7. tasks.py ===")
tasks_text = read(ROOT / "environment/tasks.py")

check("task_easy"   in tasks_text, "task_easy defined")
check("task_medium" in tasks_text, "task_medium defined")
check("task_hard"   in tasks_text, "task_hard defined")
check("TASK_REGISTRY" in tasks_text, "TASK_REGISTRY dict defined")
check("get_task_easy"   in tasks_text, "get_task_easy() function")
check("get_task_medium" in tasks_text, "get_task_medium() function")
check("get_task_hard"   in tasks_text, "get_task_hard() function")

# Check realistic patient counts
p_counts = tasks_text.count("patient_id=")
check(p_counts >= 14, f"Enough patients defined ({p_counts} total across tasks)",
      f"only {p_counts}")


# ─────────────────────────────────────────────
# 8. graders.py validation
# ─────────────────────────────────────────────

print("\n=== 8. graders.py ===")
graders_text = read(ROOT / "environment/graders.py")

check("grade_easy("   in graders_text, "grade_easy() function")
check("grade_medium(" in graders_text, "grade_medium() function")
check("grade_hard("   in graders_text, "grade_hard() function")
check("GRADER_REGISTRY" in graders_text, "GRADER_REGISTRY dict")
check("class GradeResult" in graders_text, "GradeResult class")

# Check score is bounded
check("max(0.0, min(1.0," in graders_text, "Score clamped to [0.0, 1.0]")

# Check each grader returns GradeResult
check(graders_text.count("return GradeResult(") >= 3,
      "All 3 graders return GradeResult")

# Check partial credit (different weights per task)
check("0.60" in graders_text and "0.40" in graders_text and "0.35" in graders_text,
      "Different weight profiles across tasks (partial credit)")

# Check ignored critical penalty
check("ignored_critical" in graders_text, "Ignored-critical penalty implemented")

# Check grader score never constant
check("_score_treatments(" in graders_text, "Treatment scoring is computed")
check("_score_priorities(" in graders_text, "Priority scoring is computed")


# ─────────────────────────────────────────────
# 9. server.py validation
# ─────────────────────────────────────────────

print("\n=== 9. server.py ===")
server_text = read(ROOT / "server.py")

check('"/reset"'  in server_text, "POST /reset endpoint")
check('"/step"'   in server_text, "POST /step endpoint")
check('"/state"'  in server_text, "GET /state endpoint")
check('"/health"' in server_text, "GET /health endpoint")
check("env.reset()" in server_text, "Calls env.reset()")
check("env.step("   in server_text, "Calls env.step()")
check("model_dump()" in server_text, "Returns model_dump() for serialization")

check("from fastapi import" in server_text, "Uses FastAPI")
check("CORSMiddleware" in server_text,      "CORS middleware enabled")


# ─────────────────────────────────────────────
# 10. Dockerfile validation
# ─────────────────────────────────────────────

print("\n=== 10. Dockerfile ===")
docker_text = read(ROOT / "Dockerfile")

check("FROM python:3.11" in docker_text,   "Uses Python 3.11 base image")
check("EXPOSE 7860"      in docker_text,   "Exposes port 7860 (HuggingFace)")
check("appuser"          in docker_text,   "Non-root user created")
check("USER appuser"     in docker_text,   "Runs as non-root user")
check("HEALTHCHECK"      in docker_text,   "HEALTHCHECK defined")
check("/health"          in docker_text,   "Healthcheck uses /health endpoint")
check("uvicorn"          in docker_text,   "uvicorn server command")
check("--workers 1"      in docker_text,   "Single worker (2vCPU/8GB compat)")
check("COPY requirements.txt" in docker_text, "requirements.txt copied")
check("COPY environment/" in docker_text,  "environment/ directory copied")


# ─────────────────────────────────────────────
# 11. requirements.txt
# ─────────────────────────────────────────────

print("\n=== 11. requirements.txt ===")
req_text = read(ROOT / "requirements.txt")

check("pydantic" in req_text,   "pydantic in requirements")
check("fastapi"  in req_text,   "fastapi in requirements")
check("uvicorn"  in req_text,   "uvicorn in requirements")
check("openai"   in req_text,   "openai in requirements")


# ─────────────────────────────────────────────
# 12. README.md
# ─────────────────────────────────────────────

print("\n=== 12. README.md ===")
readme_text = read(ROOT / "README.md")

required_sections = [
    "Problem Description",
    "State Space",
    "Action Space",
    "Reward",
    "Grader",
    "Task",
    "Setup",
    "inference.py",
    "Docker",
    "Baseline",
]
for section in required_sections:
    check(section in readme_text, f"README contains '{section}' section")

check("[START] task:" in readme_text, "README shows correct [START] log format")
check("[STEP] observation:" in readme_text, "README shows correct [STEP] log format")
check("[END] score:" in readme_text, "README shows correct [END] log format")


# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"VALIDATION SUMMARY")
print(f"{'='*60}")
print(f"Checks run : {checks_run}")
print(f"Passed     : {checks_run - len(errors) - len(warnings)}")
print(f"Warnings   : {len(warnings)}")
print(f"Errors     : {len(errors)}")

if warnings:
    print(f"\nWarnings:")
    for w in warnings:
        print(f"  {WARN} {w}")

if errors:
    print(f"\nErrors (must fix):")
    for e in errors:
        print(f"  {FAIL}  {e}")
    print(f"\n{FAIL}  VALIDATION FAILED — {len(errors)} error(s) must be fixed.")
    sys.exit(1)
else:
    print(f"\n{PASS}  ALL CHECKS PASSED — project is ready for submission.")
    sys.exit(0)
