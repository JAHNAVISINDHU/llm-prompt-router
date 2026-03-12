"""
Tests for the LLM-Powered Prompt Router.
Run: pytest test_router.py -v
"""

import json
import os
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path


# ─── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def set_dummy_api_key(monkeypatch):
    """Ensure OPENAI_API_KEY is set for all tests."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy-key-for-unit-tests")


def make_mock_client(content: str):
    """Helper: build a minimal OpenAI client mock that returns `content`."""
    mock_message = MagicMock()
    mock_message.content = content
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


# ─── Prompt config tests ───────────────────────────────────────────────────

def test_prompts_file_exists():
    from router import PROMPTS_FILE
    assert PROMPTS_FILE.exists(), "prompts.json must exist"


def test_at_least_four_expert_prompts():
    from router import PROMPTS, VALID_INTENTS
    assert len(VALID_INTENTS) >= 4, "Must define at least 4 expert intents"


def test_each_prompt_has_system_prompt_key():
    from router import PROMPTS
    for intent, config in PROMPTS.items():
        assert "system_prompt" in config, f"Intent '{intent}' missing 'system_prompt'"
        assert len(config["system_prompt"]) > 20, f"Intent '{intent}' system_prompt too short"


def test_expected_intents_present():
    from router import PROMPTS
    for expected in ("code", "data_analysis", "writing", "career", "unclear"):
        assert expected in PROMPTS, f"Expected intent '{expected}' not found in prompts.json"


# ─── classify_intent tests ─────────────────────────────────────────────────

def test_classify_intent_returns_correct_schema():
    from router import classify_intent
    client = make_mock_client('{"intent": "code", "confidence": 0.95}')
    result = classify_intent("How do I reverse a string in Python?", client)
    assert "intent" in result
    assert "confidence" in result
    assert isinstance(result["intent"], str)
    assert isinstance(result["confidence"], float)


def test_classify_intent_valid_code():
    from router import classify_intent
    client = make_mock_client('{"intent": "code", "confidence": 0.92}')
    result = classify_intent("Fix my Python function", client)
    assert result["intent"] == "code"
    assert result["confidence"] == pytest.approx(0.92)


def test_classify_intent_valid_writing():
    from router import classify_intent
    client = make_mock_client('{"intent": "writing", "confidence": 0.88}')
    result = classify_intent("Help me write a cover letter", client)
    assert result["intent"] == "writing"


def test_classify_intent_malformed_json_defaults_to_unclear():
    """Req 6: malformed JSON must not crash and must default to unclear/0.0."""
    from router import classify_intent
    client = make_mock_client("Sorry, I can't classify that right now.")
    result = classify_intent("some message", client)
    assert result["intent"] == "unclear"
    assert result["confidence"] == 0.0


def test_classify_intent_empty_string_defaults_to_unclear():
    from router import classify_intent
    client = make_mock_client("")
    result = classify_intent("test", client)
    assert result["intent"] == "unclear"
    assert result["confidence"] == 0.0


def test_classify_intent_partial_json_defaults_to_unclear():
    from router import classify_intent
    client = make_mock_client('{"intent": "code"')  # missing closing brace
    result = classify_intent("debug this", client)
    assert result["intent"] == "unclear"
    assert result["confidence"] == 0.0


def test_classify_intent_unknown_intent_normalised_to_unclear():
    """If the LLM invents an intent not in our list, normalise to unclear."""
    from router import classify_intent
    client = make_mock_client('{"intent": "astrology", "confidence": 0.99}')
    result = classify_intent("what does Mars mean?", client)
    assert result["intent"] == "unclear"
    assert result["confidence"] == 0.0


def test_classify_intent_strips_markdown_fences():
    """LLM sometimes wraps JSON in ```json ... ``` — should still parse."""
    from router import classify_intent
    client = make_mock_client('```json\n{"intent": "career", "confidence": 0.85}\n```')
    result = classify_intent("How do I negotiate salary?", client)
    assert result["intent"] == "career"
    assert result["confidence"] == pytest.approx(0.85)


def test_classify_intent_api_exception_defaults_to_unclear():
    """If the API call itself throws, default to unclear."""
    from router import classify_intent
    client = MagicMock()
    client.chat.completions.create.side_effect = Exception("Network error")
    result = classify_intent("test message", client)
    assert result["intent"] == "unclear"
    assert result["confidence"] == 0.0


# ─── route_and_respond tests ───────────────────────────────────────────────

def test_route_and_respond_unclear_returns_clarification_question():
    """Req 4: unclear intent must return a question, not route to an expert."""
    from router import route_and_respond
    client = MagicMock()  # should NOT be called for unclear intent
    result = route_and_respond("???", {"intent": "unclear", "confidence": 0.0}, client)
    assert isinstance(result, str)
    assert "?" in result  # must be a question
    client.chat.completions.create.assert_not_called()


def test_route_and_respond_unclear_mentions_supported_intents():
    """Clarification response should guide user toward supported intents."""
    from router import route_and_respond
    client = MagicMock()
    result = route_and_respond("huh", {"intent": "unclear", "confidence": 0.1}, client)
    result_lower = result.lower()
    assert any(word in result_lower for word in ["coding", "code", "data", "writing", "career"])


def test_route_and_respond_code_calls_llm():
    """For a known intent, the LLM should be called."""
    from router import route_and_respond
    client = make_mock_client("Here is how you do it in Python...")
    result = route_and_respond("How do I sort a list?", {"intent": "code", "confidence": 0.9}, client)
    assert "Python" in result or len(result) > 0
    client.chat.completions.create.assert_called_once()


def test_route_and_respond_uses_correct_system_prompt():
    """The system prompt used must match the classified intent."""
    from router import route_and_respond, PROMPTS
    client = make_mock_client("Response text")
    route_and_respond("Help me write an email", {"intent": "writing", "confidence": 0.9}, client)
    call_kwargs = client.chat.completions.create.call_args
    messages = call_kwargs[1]["messages"] if call_kwargs[1] else call_kwargs[0][0]
    system_msg = next(m for m in messages if m["role"] == "system")
    assert system_msg["content"] == PROMPTS["writing"]["system_prompt"]


def test_route_and_respond_returns_string():
    from router import route_and_respond
    client = make_mock_client("Some expert response")
    result = route_and_respond("Explain linear regression", {"intent": "data_analysis", "confidence": 0.88}, client)
    assert isinstance(result, str)


# ─── Logging tests ─────────────────────────────────────────────────────────

def test_log_request_creates_file(tmp_path, monkeypatch):
    """Req 5: log entries are written to route_log.jsonl."""
    import router
    test_log = tmp_path / "route_log.jsonl"
    monkeypatch.setattr(router, "LOG_FILE", test_log)

    router.log_request(
        user_message="Test message",
        classified={"intent": "code", "confidence": 0.9},
        final_response="Here is some code",
    )

    assert test_log.exists()
    lines = test_log.read_text().strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["intent"] == "code"
    assert entry["confidence"] == pytest.approx(0.9)
    assert entry["user_message"] == "Test message"
    assert entry["final_response"] == "Here is some code"


def test_log_request_appends_multiple_entries(tmp_path, monkeypatch):
    import router
    test_log = tmp_path / "route_log.jsonl"
    monkeypatch.setattr(router, "LOG_FILE", test_log)

    for i in range(3):
        router.log_request(
            user_message=f"Message {i}",
            classified={"intent": "writing", "confidence": 0.8},
            final_response=f"Response {i}",
        )

    lines = test_log.read_text().strip().splitlines()
    assert len(lines) == 3
    for line in lines:
        entry = json.loads(line)
        assert "intent" in entry
        assert "confidence" in entry
        assert "user_message" in entry
        assert "final_response" in entry


def test_log_entry_has_all_required_keys(tmp_path, monkeypatch):
    """Req 5: each entry must contain intent, confidence, user_message, final_response."""
    import router
    test_log = tmp_path / "route_log.jsonl"
    monkeypatch.setattr(router, "LOG_FILE", test_log)
    router.log_request("Q", {"intent": "career", "confidence": 0.75}, "A")
    entry = json.loads(test_log.read_text().strip())
    for key in ("intent", "confidence", "user_message", "final_response"):
        assert key in entry, f"Missing required key: {key}"


# ─── Integration-style test ────────────────────────────────────────────────

def test_process_message_full_pipeline(tmp_path, monkeypatch):
    """End-to-end: process_message should classify, route, log, and return result."""
    import router
    test_log = tmp_path / "route_log.jsonl"
    monkeypatch.setattr(router, "LOG_FILE", test_log)

    # First call → classifier returns code intent
    # Second call → expert returns a response
    classifier_mock = MagicMock()
    classifier_mock.content = '{"intent": "code", "confidence": 0.93}'

    expert_mock = MagicMock()
    expert_mock.content = "Here is a Python solution..."

    call_count = [0]

    def side_effect(**kwargs):
        call_count[0] += 1
        mock_response = MagicMock()
        if call_count[0] == 1:
            mock_response.choices = [MagicMock(message=classifier_mock)]
        else:
            mock_response.choices = [MagicMock(message=expert_mock)]
        return mock_response

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = side_effect

    with patch("router.get_client", return_value=mock_client):
        result = router.process_message("How do I write a binary search?")

    assert result["intent"] == "code"
    assert result["confidence"] == pytest.approx(0.93)
    assert "Python" in result["final_response"]
    assert test_log.exists()
