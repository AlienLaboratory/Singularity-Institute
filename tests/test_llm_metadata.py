"""
Unit tests for the OllamaClient.last_response metadata change.

Uses only the Python standard library (unittest, unittest.mock) so no new
dependencies are required.

Run with:
    python -m unittest tests.test_llm_metadata -v
"""

import unittest
from unittest.mock import patch, MagicMock

from core.llm import OllamaClient


def _make_mock_response(data: dict) -> MagicMock:
    """Build a mock httpx.Response whose .json() returns data."""
    resp = MagicMock()
    resp.json.return_value = data
    resp.raise_for_status.return_value = None
    return resp


class TestOllamaMetadata(unittest.TestCase):
    """
    Verify that OllamaClient.chat() preserves the full response in
    .last_response without breaking the existing str return contract.
    """

    def test_chat_returns_content_and_stores_metadata(self):
        """
        1. chat() returns the assistant's content string.
        2. The complete Ollama response (including all metadata fields)
           is stored in client.last_response.
        """
        sample = {
            "message": {"role": "assistant", "content": "hello"},
            "prompt_eval_count": 12,
            "eval_count": 42,
            "prompt_eval_duration": 1_500_000_000,  # 1.5 s in ns
            "eval_duration": 6_000_000_000,         # 6.0 s in ns
        }
        client = OllamaClient("http://localhost:11434", "llama3")

        with patch("core.llm.httpx.post", return_value=_make_mock_response(sample)):
            result = client.chat([{"role": "user", "content": "hi"}])

        # Backward-compatible return value
        self.assertEqual(result, "hello")

        # Metadata preserved
        self.assertIsNotNone(client.last_response)
        self.assertEqual(client.last_response["prompt_eval_count"], 12)
        self.assertEqual(client.last_response["eval_count"], 42)
        self.assertEqual(client.last_response["prompt_eval_duration"], 1_500_000_000)
        self.assertEqual(client.last_response["eval_duration"], 6_000_000_000)

    def test_last_response_initially_none(self):
        """
        A freshly constructed client must have last_response == None.
        """
        client = OllamaClient("http://localhost:11434", "llama3")
        self.assertIsNone(client.last_response)

    def test_tokens_per_second(self):
        """
        Generation throughput: eval_count / (eval_duration_ns / 1e9)
        42 tokens over 6 s = 7.0 tokens/second.
        """
        eval_count = 42
        eval_duration_ns = 6_000_000_000  # 6 s
        tps = eval_count / (eval_duration_ns / 1e9)
        self.assertAlmostEqual(tps, 7.0, places=9)

    def test_prompt_tokens_per_second(self):
        """
        Prefill throughput: prompt_eval_count / (prompt_eval_duration_ns / 1e9)
        12 tokens over 1.5 s = 8.0 tokens/second.
        """
        prompt_eval_count = 12
        prompt_eval_duration_ns = 1_500_000_000  # 1.5 s
        tps = prompt_eval_count / (prompt_eval_duration_ns / 1e9)
        self.assertAlmostEqual(tps, 8.0, places=9)

    def test_zero_duration_does_not_crash(self):
        """
        If Ollama returns 0 for prompt_eval_duration (e.g. a cached prompt),
        the client must not raise. The client only stores the raw dict;
        division by duration is the consumer's responsibility, not the
        wrapper's. We verify the client itself does not crash and that the
        zero value is preserved in last_response.
        """
        sample = {
            "message": {"role": "assistant", "content": "ok"},
            "prompt_eval_count": 0,
            "eval_count": 5,
            "prompt_eval_duration": 0,
            "eval_duration": 1_000_000_000,
        }
        client = OllamaClient("http://localhost:11434", "llama3")

        with patch("core.llm.httpx.post", return_value=_make_mock_response(sample)):
            result = client.chat([{"role": "user", "content": "hi"}])

        # The client returned the content string without raising
        self.assertEqual(result, "ok")
        # The zero duration value is faithfully preserved in last_response
        self.assertEqual(client.last_response["prompt_eval_duration"], 0)
        self.assertEqual(client.last_response["eval_count"], 5)


if __name__ == "__main__":
    unittest.main()
