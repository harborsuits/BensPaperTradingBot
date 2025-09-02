import os
import requests
from typing import Optional, Dict, Any

class LLMAdapterError(Exception):
    pass

class LLMAdapter:
    """
    Unified adapter for LLM APIs (OpenAI, Anthropic Claude).
    """
    def __init__(self, provider: str = "openai", openai_key: Optional[str] = None, claude_key: Optional[str] = None):
        self.provider = provider.lower()
        self.openai_key = openai_key or os.getenv("OPENAI_API_KEY")
        self.claude_key = claude_key or os.getenv("CLAUDE_API_KEY")
        if self.provider == "openai" and not self.openai_key:
            raise LLMAdapterError("OpenAI API key not provided.")
        if self.provider == "claude" and not self.claude_key:
            raise LLMAdapterError("Claude API key not provided.")

    def generate_completion(self, prompt: str, **kwargs) -> Dict[str, Any]:
        if self.provider == "openai":
            return self._openai_completion(prompt, **kwargs)
        elif self.provider == "claude":
            return self._claude_completion(prompt, **kwargs)
        else:
            raise LLMAdapterError(f"Unknown provider: {self.provider}")

    def _openai_completion(self, prompt: str, model: str = "gpt-4", max_tokens: int = 256, temperature: float = 0.7, **kwargs) -> Dict[str, Any]:
        url = "https://api.openai.com/v1/completions"
        headers = {"Authorization": f"Bearer {self.openai_key}"}
        data = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        try:
            resp = requests.post(url, headers=headers, json=data, timeout=20)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            raise LLMAdapterError(f"OpenAI completion failed: {e}")

    def _claude_completion(self, prompt: str, model: str = "claude-2", max_tokens: int = 256, temperature: float = 0.7, **kwargs) -> Dict[str, Any]:
        url = "https://api.anthropic.com/v1/complete"
        headers = {"x-api-key": self.claude_key, "anthropic-version": "2023-06-01"}
        data = {
            "model": model,
            "prompt": prompt,
            "max_tokens_to_sample": max_tokens,
            "temperature": temperature
        }
        try:
            resp = requests.post(url, headers=headers, json=data, timeout=20)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            raise LLMAdapterError(f"Claude completion failed: {e}")
