# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Contract Validation Environment Client."""

from typing import Dict, Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import ContractValidationAction, ContractValidationObservation


class ContractValidationEnv(
    EnvClient[ContractValidationAction, ContractValidationObservation, State]
):
    """Client for the Contract Validation Environment."""

    def _step_payload(self, action: ContractValidationAction) -> Dict[str, Any]:
        # 1. Removed action.message and added action.submit_final
        return {
            "clause_id": action.clause_id,
            "risk_type": action.risk_type,
            "submit_final": action.submit_final,
            "explanation": action.explanation,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[ContractValidationObservation]:
        obs_data = payload.get("observation", {})

        # 2. Removed hardcoded echo variables.
        # Dynamically unpack the data straight into your strict Pydantic model.
        observation = ContractValidationObservation(**obs_data)

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
