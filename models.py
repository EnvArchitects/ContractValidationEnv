# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from openenv.core.env_server.types import Action, Observation
from pydantic import Field
from typing import Dict, List, Any


class ContractValidationAction(Action):
    clause_id: int = Field(
        default=0, description="The ID of the clause being reviewed (0 if submitting final)")
    risk_type: str = Field(
        default="none", description="The type of risk identified (e.g., liability, payment, termination, confidentiality, none)")
    submit_final: bool = Field(
        default=False, description="Set to True to finish the review and receive the final grade")
    explanation: str = Field(
        default="", description="Explanation of the decision")


class ContractValidationObservation(Observation):
    task_level: str = Field(
        default="easy", description="Difficulty level of the current task")
    contract_clauses: List[Dict[str, Any]] = Field(
        default_factory=list, description="List of clauses in the contract")
    flagged_risks: Dict[int, str] = Field(
        default_factory=dict, description="Risks currently flagged by the agent")
    step_count: int = Field(default=0, description="Number of steps taken")
    reward: float = Field(
        default=0.0, description="Reward delta for the current step")
    done: bool = Field(
        default=False, description="Whether the episode has ended")
    info: Dict[str, Any] = Field(
        default_factory=dict, description="Additional info, including the grader score")
