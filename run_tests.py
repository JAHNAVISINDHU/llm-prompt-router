"""
Offline unit tests using stdlib unittest + mock only.
Run: python3 run_tests.py
(No external packages required)
"""

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent))

os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy")


def make_mock_client(content: str):
    mock_message = MagicMock()
    mock_message.content = content
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


# ── Tests ────────────────────────────────────────────────────────────────────

class TestPromptsConfig(unittest.TestCase):

    def setUp(self):
        from router import PROMPTS, VALID_INTENTS, PROMPTS_FILE
        self.PROMPTS = PROMPTS
        self.VALID_INTENTS = VALID_INTENTS
        self.PROMPTS_FILE = PROMPTS_FILE

    def test_prompts_file_exists(self):
        self.assertTrue(self.PROMPTS_FILE.exists())

    def test_at_least_four_expert_intents(self):
        self.assertGreaterEqual(len(self.VALID_INTENTS), 4)

    def test_each_prompt_has_system_prompt(self):
        for intent, cfg in self.PROMPTS.items():
            self.assertIn("system_prompt", cfg, f"{intent} missing system_prompt")
            self.assertGreater(len(cfg["system_prompt"]), 20)

    def test_required_intents_present(self):
        for intent in ("code", "data_analysis", "writing", "career", "unclear"):
            self.assertIn(intent, self.PROMPTS)


class TestClassifyIntent(unittest.TestCase):

    def setUp(self):
        os.environ["OPENAI_API_KEY"] = "sk-test-dummy"

    def _classify(self, raw_content):
        from router import classify_intent
        return classify_intent("test message", make_mock_client(raw_content))

    def test_returns_intent_and_confidence(self):
        result = self._classify('{"intent": "code", "confidence": 0.95}')
        self.assertIn("intent", result)
        self.assertIn("confidence", result)

    def test_valid_code_intent(self):
        result = self._classify('{"intent": "code", "confidence": 0.9}')
        self.assertEqual(result["intent"], "code")
        self.assertAlmostEqual(result["confidence"], 0.9)

    def test_malformed_json_defaults_to_unclear(self):
        result = self._classify("I cannot classify this.")
        self.assertEqual(result["intent"], "unclear")
        self.assertEqual(result["confidence"], 0.0)

    def test_empty_response_defaults_to_unclear(self):
        result = self._classify("")
        self.assertEqual(result["intent"], "unclear")
        self.assertEqual(result["confidence"], 0.0)

    def test_partial_json_defaults_to_unclear(self):
        result = self._classify('{"intent": "code"')
        self.assertEqual(result["intent"], "unclear")
        self.assertEqual(result["confidence"], 0.0)

    def test_unknown_intent_normalised_to_unclear(self):
        result = self._classify('{"intent": "astrology", "confidence": 0.99}')
        self.assertEqual(result["intent"], "unclear")

    def test_strips_markdown_fences(self):
        result = self._classify('```json\n{"intent": "career", "confidence": 0.85}\n```')
        self.assertEqual(result["intent"], "career")
        self.assertAlmostEqual(result["confidence"], 0.85)

    def test_api_exception_defaults_to_unclear(self):
        from router import classify_intent
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("Network error")
        result = classify_intent("test", client)
        self.assertEqual(result["intent"], "unclear")
        self.assertEqual(result["confidence"], 0.0)


class TestRouteAndRespond(unittest.TestCase):

    def test_unclear_returns_clarification_without_llm_call(self):
        from router import route_and_respond
        client = MagicMock()
        result = route_and_respond("???", {"intent": "unclear", "confidence": 0.0}, client)
        self.assertIsInstance(result, str)
        self.assertIn("?", result)
        client.chat.completions.create.assert_not_called()

    def test_unclear_mentions_supported_intents(self):
        from router import route_and_respond
        client = MagicMock()
        result = route_and_respond("huh", {"intent": "unclear", "confidence": 0.0}, client)
        result_lower = result.lower()
        self.assertTrue(any(w in result_lower for w in ["coding", "code", "data", "writing", "career"]))

    def test_known_intent_calls_llm(self):
        from router import route_and_respond
        client = make_mock_client("Here's the answer")
        route_and_respond("help", {"intent": "code", "confidence": 0.9}, client)
        client.chat.completions.create.assert_called_once()

    def test_uses_correct_system_prompt(self):
        from router import route_and_respond, PROMPTS
        client = make_mock_client("Response")
        route_and_respond("write email", {"intent": "writing", "confidence": 0.9}, client)
        call_kwargs = client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        system_msg = next(m for m in messages if m["role"] == "system")
        self.assertEqual(system_msg["content"], PROMPTS["writing"]["system_prompt"])

    def test_returns_string(self):
        from router import route_and_respond
        client = make_mock_client("Some response")
        result = route_and_respond("analyse data", {"intent": "data_analysis", "confidence": 0.8}, client)
        self.assertIsInstance(result, str)


class TestLogging(unittest.TestCase):

    def test_creates_jsonl_file(self):
        import tempfile, router
        with tempfile.TemporaryDirectory() as tmpdir:
            test_log = Path(tmpdir) / "route_log.jsonl"
            original = router.LOG_FILE
            router.LOG_FILE = test_log
            try:
                router.log_request("Test msg", {"intent": "code", "confidence": 0.9}, "Response")
                self.assertTrue(test_log.exists())
                entry = json.loads(test_log.read_text().strip())
                self.assertEqual(entry["intent"], "code")
                self.assertAlmostEqual(entry["confidence"], 0.9)
                self.assertEqual(entry["user_message"], "Test msg")
                self.assertEqual(entry["final_response"], "Response")
            finally:
                router.LOG_FILE = original

    def test_appends_multiple_entries(self):
        import tempfile, router
        with tempfile.TemporaryDirectory() as tmpdir:
            test_log = Path(tmpdir) / "route_log.jsonl"
            original = router.LOG_FILE
            router.LOG_FILE = test_log
            try:
                for i in range(3):
                    router.log_request(f"Msg {i}", {"intent": "writing", "confidence": 0.8}, f"Resp {i}")
                lines = test_log.read_text().strip().splitlines()
                self.assertEqual(len(lines), 3)
                for line in lines:
                    entry = json.loads(line)
                    for key in ("intent", "confidence", "user_message", "final_response"):
                        self.assertIn(key, entry)
            finally:
                router.LOG_FILE = original


class TestProcessMessage(unittest.TestCase):

    def test_full_pipeline(self):
        import tempfile, router
        with tempfile.TemporaryDirectory() as tmpdir:
            test_log = Path(tmpdir) / "route_log.jsonl"
            original_log = router.LOG_FILE
            router.LOG_FILE = test_log

            call_count = [0]

            def side_effect(**kwargs):
                call_count[0] += 1
                mock_response = MagicMock()
                if call_count[0] == 1:
                    mock_response.choices = [MagicMock(message=MagicMock(content='{"intent": "code", "confidence": 0.93}'))]
                else:
                    mock_response.choices = [MagicMock(message=MagicMock(content="Here is a Python solution..."))]
                return mock_response

            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = side_effect

            try:
                with patch("router.get_client", return_value=mock_client):
                    result = router.process_message("How do I write binary search?")
                self.assertEqual(result["intent"], "code")
                self.assertAlmostEqual(result["confidence"], 0.93)
                self.assertIn("Python", result["final_response"])
                self.assertTrue(test_log.exists())
            finally:
                router.LOG_FILE = original_log


# ── Runner 

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [TestPromptsConfig, TestClassifyIntent, TestRouteAndRespond, TestLogging, TestProcessMessage]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
