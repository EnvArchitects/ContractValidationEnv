from openenv.core.env_server import create_fastapi_app

# 1. Import models from the root folder
from models import ContractValidationAction, ContractValidationObservation

# 2. Explicitly import the environment class from the "server" folder
from server.contract_validation_environment import ContractValidationEnvironment

# Let OpenEnv dynamically generate the FastAPI application for you.
# NOTICE: We pass ContractValidationEnvironment directly, without ()
app = create_fastapi_app(
    ContractValidationEnvironment,
    ContractValidationAction,
    ContractValidationObservation
)


def main():
    """Entry point required by OpenEnv multi-mode deployment."""
    import uvicorn
    # Tell Uvicorn to look inside the server folder for the app
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
