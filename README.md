Hospital Decision-Making OpenEnv

A simulation-based AI project that models real-time hospital emergency decision-making under resource constraints. This system evaluates how effectively an agent allocates limited medical resources to maximize patient outcomes.

Project Overview

This project simulates a hospital emergency environment where:

Patients arrive with varying severity levels
Resources (ICU beds, doctors, general beds) are limited
Decisions must be made in real time

An intelligent agent interacts with the environment to:

Prioritize patients
Allocate resources
Decide treatments

The system evaluates decisions using a reward-based mechanism.

Objectives
Simulate real-world hospital decision-making
Implement an AI agent for resource allocation
Optimize patient outcomes under constraints
Evaluate decisions using a scoring system
Technology Stack
Python 3.8+
FastAPI
Pydantic
LLM APIs / Hugging Face (optional)
Docker (optional)
Project Structure
hospital-openenv/
│
├── environment/
│   ├── env.py
│   ├── models.py
│   ├── tasks.py
│   ├── graders.py
│   └── utils.py
│
├── server.py
├── inference.py
├── openenv.yaml
├── requirements.txt
├── Dockerfile
└── README.md
System Workflow
The environment provides:
Patient data
Available resources
The agent processes input and decides:
Patient priority
Treatment strategy
Resource allocation
The environment evaluates:
Decision quality
Rewards or penalties
The simulation proceeds iteratively.
Installation Guide
1. Clone the Repository
git clone <your-repository-link>
cd hospital-openenv
2. Create Virtual Environment
python -m venv venv

Activate the environment:

Windows:

venv\Scripts\activate

Mac/Linux:

source venv/bin/activate
3. Install Dependencies
pip install -r requirements.txt
4. Environment Variables (Optional)

Create a .env file if using external APIs:

API_KEY=your_api_key_here
MODEL_NAME=mistralai/mistral-7b-instruct
API_BASE_URL=https://api.openai.com/v1
Running the Project
Run the Server
python server.py

The server will start at:

http://localhost:8000
Run the Agent
python inference.py

This executes the agent within the simulation environment.

Using uv (Optional)
uv run server.py
Example Workflow
Start the server
Run the inference script
The agent interacts with the environment
Output includes decisions, scores, and performance metrics
Evaluation Criteria

The system evaluates performance based on:

Accuracy of patient prioritization
Efficiency in resource allocation
Patient outcome optimization
Handling of emergency scenarios
Key Features
Real-time simulation environment
Resource-constrained decision-making
Reward-based evaluation system
Extensible architecture
Support for LLM-based agents
Future Enhancements
Integration with reinforcement learning algorithms
Graphical user interface
Real-time visualization dashboard
More complex patient modeling
Docker Setup (Optional)

Build the Docker image:

docker build -t hospital-env .

Run the container:

docker run -p 8000:8000 hospital-env
Common Issues and Solutions
Virtual environment not activating

Ensure the correct activation command is used:

venv\Scripts\activate
Module not found errors

Install dependencies again:

pip install -r requirements.txt
API-related errors

Verify:

API key
Base URL
Model name
Use Cases
AI applications in healthcare
Resource allocation optimization
Reinforcement learning environments
Decision support systems
Author
Name: Sharat Patil and Deepak Bhat
Project: Hospital Decision-Making OpenEnv
Course: Engineering Project
Conclusion

This project demonstrates the application of artificial intelligence in critical decision-making scenarios, particularly in healthcare systems where efficient resource allocation is essential for improving patient outcomes.