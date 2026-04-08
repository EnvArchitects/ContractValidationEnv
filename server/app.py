from openenv.core.env_server import create_fastapi_app

# 1. Import models from the root folder
from models import ContractValidationAction, ContractValidationObservation

# 2. Explicitly import the environment from the "server" folder
from server.contract_validation_environment import ContractValidationEnvironment

# Instantiate your environment
env = ContractValidationEnvironment()

# Let OpenEnv dynamically generate the FastAPI application for you
app = create_fastapi_app(env, ContractValidationAction,
                         ContractValidationObservation)


def main():
    """Entry point required by OpenEnv multi-mode deployment."""
    import uvicorn
    # Tell Uvicorn to look inside the server folder for the app
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
