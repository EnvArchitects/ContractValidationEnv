# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI application for the Contract Validation Environment."""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import ContractValidationAction, ContractValidationObservation
    from .contract_validation_environment import ContractValidationEnvironment
except (ModuleNotFoundError, ImportError):
    from models import ContractValidationAction, ContractValidationObservation
    from server.contract_validation_environment import ContractValidationEnvironment

app = create_app(
    ContractValidationEnvironment,
    ContractValidationAction,
    ContractValidationObservation,
    env_name="contract_validation",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    main()
