from contract_validation_environment import ContractValidationEnvironment
from openenv.core.env_server import EnvServer
from fastapi import FastAPI
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


# Now we can safely import from the root folder and the local server folder

# Initialize the FastAPI application
app = FastAPI(title="Contract Validation Environment API")

# Bind the OpenEnv server logic to the FastAPI app
server = EnvServer(app, ContractValidationEnvironment)


def main():
    """Entry point required by OpenEnv multi-mode deployment."""
    import uvicorn
    # This allows you to run `python app.py` directly for local testing
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
