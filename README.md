 Hospital Decision-Making OpenEnv

This project simulates a hospital emergency department where decisions actually matter.

The goal is simple:
given a set of patients and limited hospital resources, your system needs to decide:

who gets treated first
what treatment to give
how to use ICU beds, doctors, and general beds efficiently

Bad decisions cost points. Good decisions save them.

 What this project does

At every step, the environment gives you:

a list of patients (with symptoms, severity, urgency)
available resources (ICU beds, doctors, etc.)

Your job:

assign a treatment to each patient
decide resource allocation
rank them by priority

Then the system:

calculates a reward
evaluates your decisions
updates the environment
Why this is interesting

This isn’t a toy problem.

You’ll face:

limited ICU beds
multiple critical patients
time pressure (patients worsen over time)
trade-offs between patients

In the hard task, things get worse:

more patients than resources
and even a new emergency patient appears mid-run
 Project Structure
hospital-openenv/
├── environment/
│   ├── env.py        # main environment logic
│   ├── models.py     # data structures
│   ├── tasks.py      # predefined scenarios
│   ├── graders.py    # evaluation logic
│   └── utils.py      # helper functions
├── server.py         # FastAPI server
├── inference.py      # agent runner (LLM-based)
├── openenv.yaml      # environment config
├── Dockerfile        # container setup
├── requirements.txt
└── README.md