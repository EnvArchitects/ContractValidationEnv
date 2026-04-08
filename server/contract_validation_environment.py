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
            {"id": 1, "text": "Confidential Information. Receiving Party agrees to protect Disclosing Party's Confidential Information with the same degree of care it uses to protect its own. However, Receiving Party may disclose such information to any third-party marketing affiliates without prior written consent, provided such affiliates are bound by standard non-disclosure terms."},
            {"id": 2, "text": "Severability. If any provision of this Agreement is held to be invalid or unenforceable by a court of competent jurisdiction, the remaining provisions shall continue in full force and effect without being impaired or invalidated in any way."},
            {"id": 3,
                "text": "Indemnification. Client agrees to defend and indemnify Vendor against all third-party claims arising from Client's use of the Services. Conversely, Vendor's total aggregate liability arising out of or related to this Agreement, whether in contract, tort, or otherwise, shall under no circumstances exceed the total amounts actually paid by Client in the one (1) month immediately preceding the event giving rise to the claim."},
            {"id": 4, "text": "Governing Law and Venue. This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware, exclusive of its choice of law principles. Any legal action or proceeding arising under this Agreement will be brought exclusively in the federal or state courts located in New Castle County, Delaware."},
            {"id": 5,
                "text": "Data Processing. Vendor shall process User Data solely for the purpose of providing the Services. Notwithstanding the foregoing, Vendor reserves the right to aggregate and anonymize User Data for internal analytics. Vendor explicitly disclaims any obligation to adhere to the notification provisions outlined in the California Consumer Privacy Act (CCPA) in the event of a breach involving such aggregated data."}
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

        # Reviewer fix: explicitly clamp reward to prevent negative trajectories
        reward = max(0.0, reward)

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
