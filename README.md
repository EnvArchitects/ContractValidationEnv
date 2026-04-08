---
title: Contract Validation Environment Server
emoji: 📝
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Contract Validation Environment

The **Contract Validation Environment** is an OpenEnv-compliant RL and LLM benchmark designed to test an agent's ability to act as a precise legal assistant. The agent must review various contract clauses, identify specific legal risks (e.g., liability, termination, payment), and correctly flag them without generating false positives on standard, safe clauses.

## 🚀 Motivation
Legal contract review is a massive industry, but off-the-shelf LLMs often struggle with "alert fatigue"—flagging everything as a risk. This environment challenges agents to precisely isolate genuine liabilities across varying difficulty levels while explicitly rewarding speed and accuracy.

---

## 🎯 Tasks & Difficulty

The environment features 3 deterministic tasks with increasing complexity:

1. **Easy:** 1 clause. Contains a single, explicit liability risk. Tests basic risk identification.
2. **Medium:** 3 clauses. Requires identifying payment and termination risks while actively ignoring a safe governing-law distractor clause.
3. **Hard:** 5 clauses. A complex mix of confidentiality, liability, and compliance risks interspersed with safe, standard boilerplate clauses. Challenges frontier models to avoid false positives.

---

## 📊 Environment Details

### Action Space (`ContractValidationAction`)
- `clause_id` (int): The ID of the clause being reviewed (Set to 0 if submitting final).
- `risk_type` (str): The identified risk (e.g., 'liability', 'payment', 'termination', 'confidentiality', 'compliance', or 'none').
- `submit_final` (bool): Set to `True` when the agent has finished flagging risks to end the episode and receive a final score.
- `explanation` (str): The agent's chain-of-thought or reasoning for the decision.

### Observation Space (`ContractValidationObservation`)
- `task_level` (str): Difficulty level of the current task ("easy", "medium", "hard").
- `contract_clauses` (list): List of dictionaries containing the `id` and `text` of the contract clauses to review.
- `flagged_risks` (dict): A dictionary mapping clause IDs to the risks currently flagged by the agent.
- `step_count` (int): Number of steps taken in the current episode.
- `reward` (float): The reward delta granted for the most recent action.
- `done` (bool): Whether the episode has concluded.
- `info` (dict): Additional environment info, including the current internal `score` from the grader.

### Reward Function & Grader
The environment utilizes a **trajectory-based reward system**. The grader calculates a score between `0.0` and `1.0` based on precision and recall.
* **Positive Reward:** Granted for newly correct flags.
* **Negative Penalty:** Applied for flagging safe clauses or assigning the wrong risk type.
* **Step Penalty:** A `-0.02` penalty is applied per step to encourage the agent to evaluate the contract efficiently.
* **Completion Bonus:** A `+0.5` bonus is awarded if the agent submits the contract with a perfect 1.0 grader score.

---

## 🛠️ Quick Start

The simplest way to use the environment programmatically is through the `ContractValidationEnv` client:

```python
from contract_validation import ContractValidationAction, ContractValidationEnv

try:
    # Create environment from Docker image
    env = ContractValidationEnv.from_docker_image("openenv-contract-validation:latest")

    # Reset environment to a specific task
    result = env.reset(task_level="medium")
    print(f"Clauses to review: {result.observation.contract_clauses}")

    # Take an action: Flag Clause 2 as a payment risk
    action = ContractValidationAction(
        clause_id=2, 
        risk_type="payment", 
        submit_final=False, 
        explanation="Clause dictates payment regardless of dispute."
    )
    result = env.step(action)
    print(f"Current Flags: {result.observation.flagged_risks}")
    print(f"Step Reward: {result.reward}")

    # Submit the final review
    submit_action = ContractValidationAction(clause_id=0, risk_type="none", submit_final=True)
    final_result = env.step(submit_action)
    print(f"Final Score: {final_result.observation.info['score']}")

finally:
    # Always clean up
    env.close()
```

## Project Structure

```
contract_validation/
├── .dockerignore          # Docker build exclusions
├── .env                   # Local environment variables (API keys - DO NOT COMMIT)
├── .gitignore             # Git tracking exclusions (ignores .env, caches, etc.)
├── __init__.py            # Module exports
├── README.md              # Project documentation (with tags: - openenv)
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Project metadata and dependencies
├── uv.lock                # Locked dependencies (generated)
├── client.py              # ContractValidationEnv client
├── inference.py           # Evaluation script for the OpenEnv grader (JSON logging)
├── models.py              # Action and Observation Pydantic models
├── Dockerfile             # Container image definition
└── server/
    ├── __init__.py        # Server module exports
    ├── contract_validation_environment.py  # Core environment logic and task data
    └── app.py             # FastAPI application (HTTP + WebSocket endpoints)
```
