import os
import json
import logging

try:
    from openai import OpenAI  # type: ignore
except Exception:  # ImportError or runtime issues
    OpenAI = None  # type: ignore

logger = logging.getLogger(__name__)

class LLMClient:
    """Client for LLM API interactions"""
    
    def __init__(self, api_key=None):
        """Initialize with API key"""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if OpenAI is None or not self.api_key:
            logger.warning("OpenAI library not available or API key missing. Falling back to no-op LLM client.")
            self.client = None
        else:
            self.client = OpenAI(api_key=self.api_key)
    
    def analyze_with_gpt4(self, prompt, max_tokens=1000, temperature=0.2):
        """
        Analyze text with GPT-4
        
        Args:
            prompt: Text prompt for analysis
            max_tokens: Maximum tokens in response
            temperature: Temperature parameter (0-1)
            
        Returns:
            Response text
        """
        if self.client is None:
            # Return a safe default JSON string
            fallback = {
                "bias": "neutral",
                "confidence": 0.5,
                "triggers": ["LLM unavailable"],
                "suggested_strategies": [],
                "reasoning": "OpenAI client not available in this environment"
            }
            return json.dumps(fallback)
        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in GPT-4 analysis: {str(e)}")
            # Return a safe default JSON string on error as well
            fallback = {
                "bias": "neutral",
                "confidence": 0.5,
                "triggers": ["LLM error"],
                "suggested_strategies": [],
                "reasoning": f"Error during analysis: {str(e)}"
            }
            return json.dumps(fallback)

# Create a global instance
_llm_client = None

def get_llm_client():
    """Get or create LLM client instance"""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client

def analyze_with_gpt4(prompt, max_tokens=1000, temperature=0.2):
    """Wrapper function for GPT-4 analysis"""
    client = get_llm_client()
    return client.analyze_with_gpt4(prompt, max_tokens, temperature) 