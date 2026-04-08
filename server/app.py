from contract_validation_environment import ContractValidationEnvironment
from models import ContractValidationAction, ContractValidationObservation
from openenv.core.env_server import EnvServer
from fastapi import FastAPI
import os
import sys

# --- THE ULTIMATE PATH FIX ---
# Force Python to recognize both the root folder AND the server folder
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)


# These will now resolve perfectly in the cloud

# Initialize the FastAPI application
app = FastAPI(title="Contract Validation Environment API")

# Bind the OpenEnv server logic to the FastAPI app
server = EnvServer(app, ContractValidationEnvironment)


def main():
    """Entry point required by OpenEnv multi-mode deployment."""
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
