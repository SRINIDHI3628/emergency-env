"""
inference.py
============
Baseline LLM agent for the Emergency Resource Allocation Environment.
Compatible with any OpenAI-compatible API (OpenAI, Groq, Together, HF TGI).

Usage:
    python inference.py --task easy --model gpt-4o-mini
    python inference.py --task medium
    python inference.py --task hard --verbose
"""

import os
import json
import argparse
import time
from typing import Dict, Any

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from env.environment import EmergencyEnv
from tasks import TASKS

# ─── Config from env vars ──────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.getenv("HF_TOKEN",     "")
OPENAI_KEY   = os.getenv("OPENAI_API_KEY", "")

# Initialize OpenAI client
if (HF_TOKEN or OPENAI_KEY) and OpenAI is not None:
    try:
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=HF_TOKEN or OPENAI_KEY,
        )
    except Exception as e:
        print(f"[WARN] Failed to initialize OpenAI client: {e}")
        client = None
else:
    client = None  # Will use fallback

MAX_RETRIES = 3


# ─── LLM call ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an emergency dispatch AI. Given a hospital state, 
you must decide which hospital and ambulance to assign to the current patient.

RULES:
- P1 (critical) patients NEED ICU beds — never send them to a hospital without available_icu > 0
- P2 (urgent) patients need a regular bed (available_beds > 0)
- P3 (stable) patients can go anywhere with capacity
- Always pick the ambulance that is available AND closest to the patient
- Always pick the hospital with capacity AND closest to the patient (within constraints)

Respond ONLY with a JSON object in this exact format (no explanation, no markdown):
{"hospital_id": <int>, "ambulance_id": <int>}"""


def call_llm(state: Dict) -> Dict:
    """Send state to LLM, get action back."""
    if client is None:
        print("[WARN] No API key provided — using fallback greedy agent")
        return fallback_action(state)
        
    user_msg = f"""Current emergency state:
{json.dumps(state, indent=2)}

Choose hospital_id and ambulance_id for the current_patient."""

    for attempt in range(MAX_RETRIES):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=100,
            )
            content = completion.choices[0].message.content.strip()
            # Strip markdown fences if present
            content = content.replace("```json", "").replace("```", "").strip()
            action = json.loads(content)
            return action
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[WARN] Parse error on attempt {attempt+1}: {e}")
            time.sleep(1)
        except Exception as e:
            print(f"[WARN] Request error on attempt {attempt+1}: {e}")
            time.sleep(2)

    # Fallback: pick first available hospital + ambulance
    print("[WARN] LLM failed — using fallback greedy agent")
    return fallback_action(state)


def fallback_action(state: Dict) -> Dict:
    """Simple greedy fallback: pick first available hospital & ambulance."""
    patient = state.get("current_patient", {})
    needs_icu = patient.get("needs_icu", False)

    for h in state["hospitals"]:
        if needs_icu and h["available_icu"] > 0:
            hosp_id = h["id"]
            break
        elif not needs_icu and h["available_beds"] > 0:
            hosp_id = h["id"]
            break
    else:
        hosp_id = state["hospitals"][0]["id"]

    for a in state["ambulances"]:
        if a["available"]:
            amb_id = a["id"]
            break
    else:
        amb_id = state["ambulances"][0]["id"]

    return {"hospital_id": hosp_id, "ambulance_id": amb_id}


# ─── Main runner ───────────────────────────────────────────────────────────────

def run_task(task_name: str, verbose: bool = False) -> Dict[str, Any]:
    task = TASKS[task_name]
    env = EmergencyEnv(task["config"])

    observation = env.reset()
    task_id   = task["config"]["task_id"]
    model_str = MODEL_NAME

    print(f"[START] task={task_id} env=EmergencyEnv model={model_str}")

    step_num = 0
    rewards = []
    total_reward = 0.0
    success = False

    while not env.done:
        step_num += 1

        if verbose:
            print(f"\n--- Step {step_num} ---")
            print(json.dumps(observation.model_dump(), indent=2))

        action = call_llm(observation.model_dump())
        result = env.step(action)

        reward = result["reward"].value
        done   = result["done"]
        info   = result["info"]
        observation = result["observation"]
        
        rewards.append(reward)
        total_reward += reward

        # Format action as string
        action_str = json.dumps(action)
        
        # Check for errors (simplified - no errors in this env)
        error = "null"

        print(
            f"[STEP] step={step_num} "
            f"action={action_str} "
            f"reward={reward:.2f} "
            f"done={str(done).lower()} "
            f"error={error}"
        )

    # Normalize score to [0,1]
    max_possible = len(task["config"].get("patients", [{"id":1}]))
    score = round(total_reward / max(max_possible, 1), 4)
    score = max(0.0, min(1.0, score))   # clamp

    # Check pass/fail
    grader = task["grader"]
    min_pass = grader.get("min_passing_reward", grader.get("min_passing_total_reward", 0.3))
    success = total_reward >= min_pass

    # Format rewards as comma-separated string
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(
        f"[END] success={str(success).lower()} "
        f"steps={step_num} "
        f"score={score:.3f} "
        f"rewards={rewards_str}"
    )

    return {
        "task": task_name,
        "success": success,
        "score": score,
        "steps": step_num,
        "total_reward": round(total_reward, 4),
        "rewards": rewards,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Emergency Env inference")
    parser.add_argument("--task",    default="easy",  choices=["easy", "medium", "hard"])
    parser.add_argument("--model",   default=None,    help="Override MODEL_NAME env var")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.model:
        MODEL_NAME = args.model

    result = run_task(args.task, verbose=args.verbose)
    print("\nFinal result:", json.dumps(result, indent=2))