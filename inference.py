from models import ContractValidationAction
from client import ContractValidationEnv
import os
import json
import textwrap
import asyncio
from typing import List, Optional

from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from the .env file BEFORE doing anything else
load_dotenv()

# IMPORT THE CLIENT

# --- MANDATORY ENV VARS ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

# SECURE: No hardcoded token here. It will strictly pull from your .env file!
HF_TOKEN = os.getenv("HF_TOKEN")

LOCAL_IMAGE_NAME = os.getenv(
    "LOCAL_IMAGE_NAME", "openenv-contract-validation:latest")

BENCHMARK = "contract_validation"
MAX_STEPS = 15


# --- STRICT JSON LOGGING ---
def log_start(task: str, env: str, model: str) -> None:
    log_data = {
        "event": "[START]",
        "task_id": task,
        "difficulty": task,
        "env": env,
        "model": model
    }
    print(json.dumps(log_data), flush=True)


def log_step(task: str, step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    log_data = {
        "event": "[STEP]",
        "task_id": task,
        "step": step,
        "action": action,
        # Clamp reward to prevent negative values breaking the OpenEnv grader
        "reward": max(0.0, round(reward, 2)),
        "done": done,
        "error": error
    }
    print(json.dumps(log_data), flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    log_data = {
        "event": "[END]",
        "success": success,
        "steps": steps,
        # Ensure score stays strictly within [0.0, 1.0]
        "score": max(0.0, min(1.0, round(score, 2))),
        "rewards": [max(0.0, round(r, 2)) for r in rewards]
    }
    print(json.dumps(log_data), flush=True)


async def run_task(client: OpenAI, task_level: str):
    env = await ContractValidationEnv.from_docker_image(LOCAL_IMAGE_NAME)

    try:
        result = await env.reset(task_level=task_level)
        obs = result.observation

        done = False
        error = None
        rewards: List[float] = []

        log_start(task=task_level, env=BENCHMARK, model=MODEL_NAME)

        while not done and obs.step_count < MAX_STEPS:
            system_prompt = textwrap.dedent("""
                You are a precise legal AI. Review the clauses and output valid JSON.
                Your JSON must match exactly:
                {
                  "thoughts": "your reasoning",
                  "clause_id": 1,
                  "risk_type": "liability",
                  "submit_final": false
                }
                Valid risk types: liability, payment, termination, confidentiality, compliance, none.
            """).strip()

            user_prompt = textwrap.dedent(f"""
                Current Clauses: {json.dumps(obs.contract_clauses)}
                Risks You Have ALREADY Flagged: {json.dumps(obs.flagged_risks)}

                Instructions:
                1. Identify any unflagged risks in the Current Clauses.
                2. If there is a risk you haven't flagged yet, output its clause_id and risk_type.
                3. DO NOT repeat an action. If a clause is already in your "ALREADY Flagged" list, leave it alone.
                4. CRITICAL: If you have found all the risks (or if the remaining clauses are perfectly safe), you MUST end the review by setting "submit_final": true, "clause_id": 0, and "risk_type": "none".
            """).strip()

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
                error = str(e).replace("\n", " ")
                action_str = "parse_error"
                action = ContractValidationAction(
                    clause_id=0, risk_type="none", submit_final=False)

            result = await env.step(action)
            obs = result.observation

            step_reward = result.reward if result.reward is not None else 0.0
            rewards.append(step_reward)
            done = result.done

            log_step(task=task_level, step=obs.step_count, action=action_str,
                     reward=step_reward, done=done, error=error)

        score = obs.info.get("score", 0.0)
        success = score == 1.0

        log_end(success=success, steps=obs.step_count,
                score=score, rewards=rewards)

    finally:
        try:
            await env.close()
        except Exception:
            pass


async def main():
    if not HF_TOKEN:
        print("CRITICAL WARNING: HF_TOKEN is missing! Make sure your .env file is set up correctly.")
        return  # Stop execution if there is no token

    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

    tasks = ["easy", "medium", "hard"]
    for t in tasks:
        await run_task(client, t)


if __name__ == "__main__":
    asyncio.run(main())
