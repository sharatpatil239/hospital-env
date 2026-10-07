# Hospital Decision-Making OpenEnv

A production-ready simulation-based AI benchmark that models real-time hospital emergency department decision-making under resource constraints. This system evaluates how effectively an agent triages patients, determines treatments, and allocates scarce medical resources under time pressure.

---

## Problem Description

In an emergency department, medical staff must make high-stakes triage and resource allocation decisions under uncertainty and scarcity.
Given a batch of patients and constrained hospital resources, the intelligent agent must decide:
- **Priority Ranking**: Which patient receives attention first.
- **Treatment Assignment**: The appropriate clinical intervention for each patient's condition.
- **Resource Allocation**: How to allocate limited ICU beds, general beds, and attending physicians efficiently.

Suboptimal decisions (e.g. allocating ICU beds to non-critical patients, delaying urgent care, or incorrect treatment) incur penalties, while correct, timely actions optimize patient health and hospital throughput.

---

## State Space

The environment state and agent observation are defined in `environment/models.py`:

- **Observation**:
  - `patients`: List of `Patient` models currently in the triage queue:
    - `patient_id`: Unique string identifier (e.g. `P001`).
    - `symptoms`: List of clinical symptoms (e.g. `chest_pain`, `shortness_of_breath`, `trauma`).
    - `severity`: Continuous float in `[0.0, 1.0]`.
    - `urgency`: Categorical `UrgencyLevel` (`critical`, `high`, `moderate`, `low`).
    - `age`: Integer patient age.
    - `comorbidities`: List of underlying medical conditions.
    - `hours_waiting`: Time spent waiting without care.
  - `resources`: `HospitalResources` model:
    - `icu_beds_available`: Number of available Intensive Care Unit beds.
    - `general_beds_available`: Number of available regular ward beds.
    - `doctors_available`: Number of available physicians.
  - `step`: Current simulation step.
  - `time_elapsed_hours`: Simulation elapsed time in hours.
  - `pending_patients`: Remaining patients not yet admitted.
  - `task_id`: Identifier of the active task.

---

## Action Space

The agent responds at each step with an `Action` model containing:

- `patient_actions`: A list of `PatientAction` objects, one for each patient:
  - `patient_id`: ID of the target patient.
  - `treatment`: Selected `TreatmentType` from:
    - `emergency_surgery`, `cardiac_intervention`, `icu_monitoring`, `antibiotics`, `pain_management`, `respiratory_support`, `neurological_eval`, `orthopedic_care`, `observation`, `discharge`.
  - `assign_icu_bed`: Boolean flag to assign an ICU bed.
  - `assign_general_bed`: Boolean flag to assign a general bed.
  - `assign_doctor`: Boolean flag to assign a dedicated doctor.
  - `priority_rank`: Unique 1-based integer rank (1 = highest priority).

---

## Reward

The environment computes dense, shaped rewards clamped to `[-1.0, +1.0]`:
- **Treatment Correctness**: Reward for matching guideline treatments; penalty for overtreatment or harmful treatments.
- **Resource Allocation**: Reward for necessary ICU allocation; penalty for wasting ICU beds on non-critical cases or exceeding capacity.
- **Priority Ordering**: Reward based on Spearman-style correlation with clinically optimal priority order.
- **Ignored Critical Penalty**: Severe penalty when critical or high-urgency patients are neglected.
- **Wait Time Degradation**: Progressive penalty as patient conditions deteriorate over time.

**Normalized Reward**: Computed as `(value + 1.0) / 2.0`, mapped to the range `[0.0, 1.0]`.

---

## Grader

Evaluation is handled by deterministic graders defined in `environment/graders.py`:
- `grade_easy`: Weights treatment correctness (60%), resource allocation (30%), priority ranking (10%).
- `grade_medium`: Weights treatment (40%), priority (35%), resource allocation (25%), minus ignored urgency penalty.
- `grade_hard`: Weights treatment (35%), priority (30%), resource allocation (35%), minus ignored critical penalty plus dynamic triage bonus.

Each grader returns a `GradeResult` containing a normalized score in `[0.0, 1.0]` and a detailed component breakdown.

---

## Task Definitions (Task)

Three tasks of increasing complexity are available in `environment/tasks.py`:

| Task ID | Name | Patients | ICU Beds | Doctors | Steps | Description |
|---|---|---|---|---|---|---|
| `task_easy` | Single Patient Triage | 1 | 2 | 2 | 1 | Single critical cardiac patient with ample resources. |
| `task_medium` | Multi-Patient Prioritization | 5 | 2 | 3 | 1 | 5 patients with conflicting urgency; requires optimal prioritization and ICU rationing. |
| `task_hard` | Critical Overload with Dynamic Events | 8+1 | 2 | 3 | 3 | Severe capacity constraint with dynamic emergency patient spike arriving at step 2. |

---

## Baseline

Benchmark baseline scores for default evaluator:
- **`task_easy`**: 0.95
- **`task_medium`**: 0.75
- **`task_hard`**: 0.40

---

## Setup & Installation (Setup)

### 1. Clone & Create Virtual Environment
```bash
git clone <your-repository-link>
cd hospital-openenv
python -m venv venv
```

Activate the environment:
- Windows: `venv\Scripts\activate`
- Mac/Linux: `source venv/bin/activate`

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Running the Server
```bash
python server.py
```
The server will start and listen on `http://0.0.0.0:8000`.

---

## inference.py

The baseline LLM agent runner is located in `inference.py`.

### Environment Variables
```bash
export OPENAI_API_KEY="your-key"
export MODEL_NAME="gpt-4o"
export API_BASE_URL="https://api.openai.com/v1"
```

### Running Inference
```bash
python inference.py
```

### Log Format
`inference.py` adheres strictly to the required OpenEnv execution log format:
```
[START] task: task_easy
[STEP] observation: {"task_id": "task_easy", "step": 0, "patients": [...], "resources": {...}} action: {"patient_actions": [...]} reward: 0.95
[END] score: 0.95
```

---

## Docker Setup (Docker)

Build and run using Docker:
```bash
docker build -t hospital-env .
docker run -p 7860:7860 hospital-env
```
Healthcheck is configured on `http://localhost:7860/health`.

---

## Author & Credits

- **Authors**: Sharat Patil and Venkat dhanush
- **Project**: Hospital Decision-Making OpenEnv Benchmark