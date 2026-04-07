import os
import json
from typing import List
from openai import OpenAI

# Import your local environment and models
from server.contract_validation_environment import ContractValidationEnvironment
from models import ContractValidationAction

# --- MANDATORY ENV VARS ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
BENCHMARK = "contract_validation"
MAX_STEPS = 15

# Initialize OpenAI Client
client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)


def run_task(task_level: str):
    env = ContractValidationEnvironment()
    obs = env.reset(task_level=task_level)

    done = False
    error = None
    rewards: List[float] = []

    # [START] Logging
    print(f"[START] task={task_level} env={BENCHMARK} model={MODEL_NAME}")

    while not done and obs.step_count < MAX_STEPS:
        # Prompt Engineering designed for robust JSON output across models
        system_prompt = """You are a precise legal AI. Review the clauses and output valid JSON.
Your JSON must match exactly:
{
  "thoughts": "your reasoning",
  "clause_id": 1,
  "risk_type": "liability",
  "submit_final": false
}
Valid risk types: liability, payment, termination, confidentiality, compliance, none."""

        user_prompt = f"""
Current Clauses: {json.dumps(obs.contract_clauses)}
Flagged Risks: {json.dumps(obs.flagged_risks)}

Pick ONE clause to evaluate. If you have flagged all risks, set 'submit_final' to true (clause_id: 0, risk_type: "none").
"""

        action_str = ""
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )

            raw_response = response.choices[0].message.content
            parsed = json.loads(raw_response)

            # Extract fields safely
            clause_id = int(parsed.get("clause_id", 0))
            risk_type = str(parsed.get("risk_type", "none"))
            submit_final = bool(parsed.get("submit_final", False))

            action_str = f"flag({clause_id}, '{risk_type}', submit={submit_final})"

            action = ContractValidationAction(
                clause_id=clause_id,
                risk_type=risk_type,
                submit_final=submit_final,
                explanation=parsed.get("thoughts", "")
            )
            error = None

        except Exception as e:
            # Handle parsing or API errors to prevent crash, pass empty action to env
            error = str(e).replace("\n", " ")  # Keep error on one line
            action_str = "parse_error"
            action = ContractValidationAction(
                clause_id=0, risk_type="none", submit_final=False)

        # Take environment step
        obs = env.step(action)
        rewards.append(obs.reward)
        done = obs.done

        # [STEP] Logging - STRICT FORMAT
        err_out = "null" if error is None else f"\"{error}\""
        print(
            f"[STEP] step={obs.step_count} action={action_str} reward={obs.reward:.2f} done={str(done).lower()} error={err_out}")

    # Calculate final states
    success = obs.info.get("score", 0.0) == 1.0
    score = obs.info.get("score", 0.0)
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])

    # [END] Logging - STRICT FORMAT
    print(
        f"[END] success={str(success).lower()} steps={obs.step_count} score={score:.2f} rewards={rewards_str}")


if __name__ == "__main__":
    if not HF_TOKEN:
        print("Warning: HF_TOKEN / API_KEY is not set. Inference may fail depending on endpoint.")

    tasks = ["easy", "medium", "hard"]
    for t in tasks:
        run_task(t)
