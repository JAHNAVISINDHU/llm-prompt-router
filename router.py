"""
LLM-Powered Prompt Router — Core Logic
Handles intent classification and response routing.
"""

import json
import os
import re
import datetime
from pathlib import Path
from openai import OpenAI


BASE_DIR = Path(__file__).parent
PROMPTS_FILE = BASE_DIR / "prompts.json"
LOG_FILE = BASE_DIR / "logs" / "route_log.jsonl"

LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

def load_prompts() -> dict:
    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

PROMPTS = load_prompts()
VALID_INTENTS = [k for k in PROMPTS.keys() if k != "unclear"]

def get_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is not set. "
            "Please set it before running the router."
        )
    return OpenAI(api_key=api_key)



CLASSIFIER_SYSTEM_PROMPT = f"""You are a precise intent classifier for an AI routing system.
Your ONLY job is to classify the user's message into one of these intents:
- code         : programming, debugging, software architecture, algorithms, dev tools
- data_analysis: statistics, datasets, machine learning, data visualization, SQL
- writing      : essays, emails, reports, editing, creative writing, communication
- career       : job search, resume, interviews, salary, career growth, networking
- unclear      : the message is ambiguous, off-topic, or doesn't fit any above category

Respond ONLY with a valid JSON object — no markdown, no explanation, no extra text.
The JSON must have exactly two keys:
  "intent"     : one of the five labels above (string)
  "confidence" : a float between 0.0 and 1.0 representing your certainty

Example output:
{{"intent": "code", "confidence": 0.95}}"""


def classify_intent(user_message: str, client: OpenAI = None) -> dict:
    """
    Calls the LLM classifier and returns a structured dict:
      { "intent": str, "confidence": float }

    On any parsing failure, defaults to:
      { "intent": "unclear", "confidence": 0.0 }
    """
    if client is None:
        client = get_client()

    default_result = {"intent": "unclear", "confidence": 0.0}

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,      
            max_tokens=60,
        )

        raw = response.choices[0].message.content.strip()

        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        parsed = json.loads(raw)

        intent = str(parsed.get("intent", "unclear")).lower()
        confidence = float(parsed.get("confidence", 0.0))

        all_intents = VALID_INTENTS + ["unclear"]
        if intent not in all_intents:
            intent = "unclear"
            confidence = 0.0

        return {"intent": intent, "confidence": confidence}

    except (json.JSONDecodeError, ValueError, KeyError, AttributeError) as parse_err:
        print(f"[WARN] classify_intent — parse error: {parse_err}. Defaulting to 'unclear'.")
        return default_result

    except Exception as api_err:
        print(f"[ERROR] classify_intent — API error: {api_err}. Defaulting to 'unclear'.")
        return default_result



CLARIFICATION_PROMPT = (
    "I'd love to help! Could you give me a bit more detail about what you need? "
    "For example, are you looking for help with:\n"
    "• **Coding** — debugging, writing scripts, software design\n"
    "• **Data Analysis** — statistics, datasets, machine learning\n"
    "• **Writing** — drafting emails, editing documents, creative content\n"
    "• **Career Advice** — resumes, interviews, job searching\n\n"
    "Just let me know and I'll point you to the right expert! 😊"
)


def route_and_respond(user_message: str, classified: dict, client: OpenAI = None) -> str:
    """
    Selects the appropriate system prompt based on `classified["intent"]`
    and generates a final response.

    For 'unclear' intent, returns a clarification question without a second LLM call.
    Returns the final response as a plain string.
    """
    if client is None:
        client = get_client()

    intent = classified.get("intent", "unclear")

    if intent == "unclear":
        return CLARIFICATION_PROMPT

    expert_config = PROMPTS.get(intent)
    if not expert_config:
        return CLARIFICATION_PROMPT

    system_prompt = expert_config["system_prompt"]

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.7,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"[ERROR] route_and_respond — API error: {e}")
        return f"I encountered an error generating a response. Please try again. (Error: {e})"


# Logging

def log_request(user_message: str, classified: dict, final_response: str) -> None:
    """Appends a JSON Lines entry to route_log.jsonl."""
    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "user_message": user_message,
        "intent": classified.get("intent"),
        "confidence": classified.get("confidence"),
        "final_response": final_response,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# High-level entry point

def process_message(user_message: str) -> dict:
    """
    Full pipeline:
      1. Classify intent
      2. Route and generate response
      3. Log the request
      4. Return a result dict
    """
    client = get_client()

    classified = classify_intent(user_message, client)
    final_response = route_and_respond(user_message, classified, client)
    log_request(user_message, classified, final_response)

    return {
        "user_message": user_message,
        "intent": classified["intent"],
        "confidence": classified["confidence"],
        "final_response": final_response,
    }
