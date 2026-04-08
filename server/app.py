from fastapi import FastAPI
from openenv.core.env_server import EnvServer

# 1. Import models from the root folder
from models import ContractValidationAction, ContractValidationObservation

# 2. Explicitly import the environment from the "server" folder
from server.contract_validation_environment import ContractValidationEnvironment

# Initialize the FastAPI application
app = FastAPI(title="Contract Validation Environment API")

# Bind the OpenEnv server logic to the FastAPI app
server = EnvServer(app, ContractValidationEnvironment)


def main():
    """Entry point required by OpenEnv multi-mode deployment."""
    import uvicorn
    # Tell Uvicorn to look inside the server folder for the app
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
