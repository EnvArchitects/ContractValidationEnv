# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from uuid import uuid4
from typing import Optional
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import ContractValidationAction, ContractValidationObservation
except ImportError:
    from models import ContractValidationAction, ContractValidationObservation

TASKS = {
    "easy": {
        "clauses": [{"id": 1, "text": "The vendor shall hold absolutely no liability for any damages."}],
        "ground_truth": {1: "liability"}
    },
    "medium": {
        "clauses": [
            {"id": 1, "text": "This agreement is governed by the laws of California."},
            {"id": 2, "text": "Client must pay all invoices regardless of ongoing service disputes."},
            {"id": 3, "text": "Either party may terminate this agreement with 1 hour written notice."}
        ],
        "ground_truth": {1: "none", 2: "payment", 3: "termination"}
    },
    "hard": {
        "clauses": [
            {"id": 1, "text": "Vendor retains the right to share Client data with third-party marketers."},
            {"id": 2, "text": "Standard support hours are 9 AM to 5 PM EST."},
            {"id": 3, "text": "Damages are capped at the total amount paid in the prior 12 months."},
            {"id": 4, "text": "The platform requires basic internet access to function."},
            {"id": 5, "text": "Vendor is not responsible for adhering to local data protection laws."}
        ],
        "ground_truth": {1: "confidentiality", 2: "none", 3: "liability", 4: "none", 5: "compliance"}
    }
}


class ContractValidationEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.flags = {}
        self.current_task_level = "easy"
        self._prev_score = 0.0

    def reset(self, task_level: Optional[str] = "easy") -> ContractValidationObservation:
        """Reset environment to a specific task level (easy, medium, hard)."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.flags = {}
        self._prev_score = 0.0
        self.current_task_level = task_level if task_level in TASKS else "easy"

        return ContractValidationObservation(
            task_level=self.current_task_level,
            contract_clauses=TASKS[self.current_task_level]["clauses"],
            flagged_risks={},
            step_count=0,
            reward=0.0,
            done=False,
            info={"score": 0.0, "message": "Environment reset."}
        )

    def step(self, action: ContractValidationAction) -> ContractValidationObservation:
        self._state.step_count += 1
        task = TASKS[self.current_task_level]
        gt = task["ground_truth"]

        done = action.submit_final or self._state.step_count >= 15

        # Record action if not submitting
        if not action.submit_final and action.clause_id in [c["id"] for c in task["clauses"]]:
            if action.risk_type.lower() == "none":
                self.flags.pop(action.clause_id, None)
            else:
                self.flags[action.clause_id] = action.risk_type.lower()

        # --- Grader & Trajectory Reward Logic ---
        total_risks = len([r for r in gt.values() if r != "none"])
        score = 0.0

        for cid, expected in gt.items():
            flagged = self.flags.get(cid, "none")
            if expected != "none":
                if flagged == expected:
                    score += 1.0 / total_risks  # Reward for correct flag
                elif flagged != "none":
                    score -= 0.25 / total_risks  # Penalty for wrong risk type on a risky clause
            else:
                if flagged != "none":
                    # Harsh penalty for flagging a safe clause
                    score -= 0.5 / max(total_risks, 1)

        # Normalize Grader Score between 0.0 and 1.0
        score = max(0.0, min(1.0, score))

        # Trajectory Reward: Delta from last step, minus a step penalty
        step_penalty = 0.02
        reward = (score - self._prev_score) - step_penalty
        self._prev_score = score

        if done and score == 1.0:
            reward += 0.5  # Bonus for submitting a perfect score

        info = {
            "score": score,
            "message": "Step processed." if not done else f"Episode finished. Final Score: {score:.2f}/1.0"
        }

        return ContractValidationObservation(
            task_level=self.current_task_level,
            contract_clauses=task["clauses"],
            flagged_risks=self.flags,
            step_count=self._state.step_count,
            reward=reward,
            done=done,
            info=info
        )

    @property
    def state(self) -> State:
        return self._state
