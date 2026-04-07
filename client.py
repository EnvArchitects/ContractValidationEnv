# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Contract Validation Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import ContractValidationAction, ContractValidationObservation


class ContractValidationEnv(
    EnvClient[ContractValidationAction, ContractValidationObservation, State]
):
    """Client for the Contract Validation Environment."""

    def _step_payload(self, action: ContractValidationAction) -> Dict:
        return {
            "message": action.message,
            "clause_id": action.clause_id,
            "risk_type": action.risk_type,
            "explanation": action.explanation,
        }

    def _parse_result(self, payload: Dict) -> StepResult[ContractValidationObservation]:
        obs_data = payload.get("observation", {})
        observation = ContractValidationObservation(
            contract=obs_data.get("contract", []),
            flagged_risks=obs_data.get("flagged_risks", {}),
            step_count=obs_data.get("step_count", 0),
            echoed_message=obs_data.get("echoed_message", ""),
            message_length=obs_data.get("message_length", 0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
