from server.contract_validation_environment import ContractValidationEnvironment
from models import ContractValidationAction, ContractValidationObservation
from openenv.core.env_server import create_app
import os

# 1. FORCE the web interface to turn on before OpenEnv loads
os.environ["ENABLE_WEB_INTERFACE"] = "true"


# 2. Import models from the root folder

# 3. Explicitly import the environment class

# 4. Use create_app (This builds BOTH the backend API and the Gradio frontend)
app = create_app(
    ContractValidationEnvironment,
    ContractValidationAction,
    ContractValidationObservation
)


def main():
    """Entry point required by OpenEnv multi-mode deployment."""
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
