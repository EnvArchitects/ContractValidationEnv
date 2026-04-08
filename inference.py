from models import ContractValidationAction
from client import ContractValidationEnv
import os
import json
import textwrap
import asyncio
from typing import List, Optional

from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# --- MANDATORY ENV VARS ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

BENCHMARK = "contract_validation"
MAX_STEPS = 15


# --- THE STRICT FORMATTING FIX ---
# The grader expects exact string matches, NOT JSON!
def log_start(task: str) -> None:
    print(f"[START] task={task}", flush=True)


def log_step(step: int, reward: float) -> None:
    # Reward must be numeric
    clamped_reward = max(0, round(reward, 2))
    print(f"[STEP] step={step} reward={clamped_reward}", flush=True)


def log_end(task: str, score: float, steps: int) -> None:
    # UPDATED: Score must be strictly between 0 and 1 (No 0.0, No 1.0)
    final_score = max(0.01, min(0.99, round(score, 2)))
    print(f"[END] task={task} score={final_score} steps={steps}", flush=True)


async def run_task(client: OpenAI, task_level: str):
    # Direct connection to your live, validated Space
    space_url = "https://envarchitects-contract-validation-env.hf.space"
    env = ContractValidationEnv(base_url=space_url)

    try:
        result = await env.reset(task_level=task_level)
        obs = result.observation
        done = False

        # Output the exact START string
        log_start(task=task_level)

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

                action = ContractValidationAction(
                    clause_id=clause_id,
                    risk_type=risk_type,
                    submit_final=submit_final,
                    explanation=parsed.get("thoughts", "")
                )

            except Exception as e:
                # Fallback action if the LLM hallucinated bad JSON
                action = ContractValidationAction(
                    clause_id=0, risk_type="none", submit_final=False)

            result = await env.step(action)
            obs = result.observation

            step_reward = result.reward if result.reward is not None else 0.0
            done = result.done

            # Output the exact STEP string
            log_step(step=obs.step_count, reward=step_reward)

        score = obs.info.get("score", 0.0)

        # Output the exact END string
        log_end(task=task_level, score=score, steps=obs.step_count)

    finally:
        try:
            await env.close()
        except Exception:
            pass


async def main():
    if not HF_TOKEN:
        print("CRITICAL WARNING: HF_TOKEN is missing! Make sure your .env file is set up correctly.")
        return

    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

    # Must run the 3 requested tasks
    tasks = ["easy", "medium", "hard"]
    for t in tasks:
        await run_task(client, t)


if __name__ == "__main__":
    asyncio.run(main())
