"""
Financial LLM Engine - Core interface for all LLM operations

This module provides a unified interface for interacting with different LLM providers,
implementing request batching, caching, and standardized response handling.
"""

import os
import time
import json
import logging
import hashlib
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass
import asyncio
from functools import lru_cache

import openai
import anthropic
import cohere
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Support for local models like LLaMA when available
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# Initialize logger
logger = logging.getLogger("financial_llm_engine")

class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    OPENAI_SECONDARY = "openai_secondary"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    MISTRAL = "mistral"
    GEMINI = "gemini"
    LOCAL = "local"  # For locally hosted models

@dataclass
class LLMResponse:
    """Standardized LLM response structure"""
    text: str
    model: str
    provider: LLMProvider
    tokens_used: int
    metadata: Dict[str, Any]
    
    def __repr__(self) -> str:
        return f"LLMResponse(provider={self.provider.value}, model={self.model}, tokens={self.tokens_used})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "text": self.text,
            "model": self.model,
            "provider": self.provider.value,
            "tokens_used": self.tokens_used,
            "metadata": self.metadata
        }

class FinancialLLMEngine:
    """
    Core LLM interface engine providing access to various LLM providers
    with batching, caching, and failover capabilities
    """
    
    def __init__(
        self, 
        config_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        default_provider: LLMProvider = LLMProvider.OPENAI,
        enable_caching: bool = True,
        max_cache_size: int = 1000,
        timeout: int = 30,
        debug: bool = False
    ):
        """
        Initialize the Financial LLM Engine
        
        Args:
            config_path: Path to configuration file with API keys
            cache_dir: Directory for response caching
            default_provider: Default LLM provider
            enable_caching: Whether to enable response caching
            max_cache_size: Maximum cache entries
            timeout: API request timeout in seconds
            debug: Enable debug logging
        """
        self.default_provider = default_provider
        self.enable_caching = enable_caching
        self.max_cache_size = max_cache_size
        self.timeout = timeout
        self.debug = debug
        
        # Set up logging
        logging_level = logging.DEBUG if debug else logging.INFO
        logging.basicConfig(level=logging_level)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize cache
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize clients
        self._init_clients()
        
        # Stats tracking
        self.stats = {
            "requests": 0,
            "cache_hits": 0,
            "tokens_used": 0,
            "errors": 0,
            "provider_calls": {p.value: 0 for p in LLMProvider}
        }
        
        logger.info(f"Financial LLM Engine initialized with default provider: {default_provider.value}")

    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from file or environment variables"""
        config = {}
        
        # Try loading from file
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                logger.info(f"Loaded configuration from {config_path}")
        
        # Fallback to environment variables
        for provider in LLMProvider:
            key_name = f"{provider.value.upper()}_API_KEY"
            if key_name in os.environ:
                config[provider.value] = {"api_key": os.environ[key_name]}
        
        # Validate
        if not config:
            logger.warning("No API keys found in config file or environment variables")
            
        return config

    def _init_clients(self):
        """Initialize API clients for each provider"""
        self.clients = {}
        
        # OpenAI
        if LLMProvider.OPENAI.value in self.config:
            openai.api_key = self.config[LLMProvider.OPENAI.value]["api_key"]
            self.clients[LLMProvider.OPENAI] = openai.Client(api_key=openai.api_key)
            
        # OpenAI Secondary
        if LLMProvider.OPENAI_SECONDARY.value in self.config:
            self.clients[LLMProvider.OPENAI_SECONDARY] = openai.Client(
                api_key=self.config[LLMProvider.OPENAI_SECONDARY.value]["api_key"]
            )
        
        # Anthropic
        if LLMProvider.ANTHROPIC.value in self.config:
            self.clients[LLMProvider.ANTHROPIC] = anthropic.Anthropic(
                api_key=self.config[LLMProvider.ANTHROPIC.value]["api_key"]
            )
            
        # Cohere
        if LLMProvider.COHERE.value in self.config:
            self.clients[LLMProvider.COHERE] = cohere.Client(
                api_key=self.config[LLMProvider.COHERE.value]["api_key"]
            )
        
        # Log available providers
        logger.info(f"Initialized clients for providers: {list(self.clients.keys())}")

    def _get_cache_key(self, provider: LLMProvider, model: str, prompt: str, 
                      params: Dict[str, Any]) -> str:
        """Generate a unique cache key for a request"""
        # Include provider, model, prompt and essential parameters in the hash
        key_dict = {
            "provider": provider.value,
            "model": model,
            "prompt": prompt,
            "params": {k: v for k, v in params.items() 
                     if k in ["temperature", "max_tokens", "top_p"]}
        }
        key_str = json.dumps(key_dict, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _check_cache(self, cache_key: str) -> Optional[LLMResponse]:
        """Check if a response is cached"""
        if not self.enable_caching:
            return None
            
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    self.stats["cache_hits"] += 1
                    logger.debug(f"Cache hit for key {cache_key}")
                    return LLMResponse(
                        text=data["text"],
                        model=data["model"],
                        provider=LLMProvider(data["provider"]),
                        tokens_used=data["tokens_used"],
                        metadata=data["metadata"]
                    )
            except Exception as e:
                logger.warning(f"Error reading cache: {e}")
        
        return None

    def _save_to_cache(self, cache_key: str, response: LLMResponse):
        """Save a response to cache"""
        if not self.enable_caching:
            return
            
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        try:
            with open(cache_file, 'w') as f:
                json.dump(response.to_dict(), f)
            logger.debug(f"Saved response to cache with key {cache_key}")
        except Exception as e:
            logger.warning(f"Error saving to cache: {e}")

    @retry(
        stop=stop_after_attempt(3), 
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError, 
                                     anthropic.RateLimitError, requests.exceptions.Timeout))
    )
    async def _call_openai(self, model: str, prompt: str, 
                        system_message: Optional[str] = None,
                        **kwargs) -> LLMResponse:
        """Call OpenAI API"""
        provider_key = LLMProvider.OPENAI
        client = self.clients.get(provider_key)
        
        if not client:
            raise ValueError(f"OpenAI client not initialized. Check API key configuration.")
        
        try:
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})
            
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                timeout=self.timeout,
                **kwargs
            )
            
            tokens_used = response.usage.total_tokens
            self.stats["tokens_used"] += tokens_used
            
            return LLMResponse(
                text=response.choices[0].message.content,
                model=model,
                provider=provider_key,
                tokens_used=tokens_used,
                metadata={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "finish_reason": response.choices[0].finish_reason
                }
            )
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Error calling OpenAI: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3), 
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((anthropic.RateLimitError, requests.exceptions.Timeout))
    )
    async def _call_anthropic(self, model: str, prompt: str, 
                           system_message: Optional[str] = None,
                           **kwargs) -> LLMResponse:
        """Call Anthropic API"""
        provider_key = LLMProvider.ANTHROPIC
        client = self.clients.get(provider_key)
        
        if not client:
            raise ValueError(f"Anthropic client not initialized. Check API key configuration.")
        
        try:
            messages = []
            if system_message:
                kwargs["system"] = system_message
            
            response = client.messages.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.pop("max_tokens", 1000),
                **kwargs
            )
            
            # Anthropic doesn't provide token counts directly in the same way,
            # so we'll estimate based on a standard conversion rate
            text = response.content[0].text
            estimated_tokens = len(text.split()) * 1.3
            
            return LLMResponse(
                text=text,
                model=model,
                provider=provider_key,
                tokens_used=int(estimated_tokens),
                metadata={
                    "stop_reason": response.stop_reason,
                    "model": response.model
                }
            )
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Error calling Anthropic: {e}")
            raise

    async def generate(
        self, 
        prompt: str,
        provider: Optional[LLMProvider] = None,
        model: Optional[str] = None,
        system_message: Optional[str] = None,
        fallback_providers: Optional[List[LLMProvider]] = None,
        cache: bool = True,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text using the specified LLM provider
        
        Args:
            prompt: The prompt to send to the LLM
            provider: LLM provider to use (defaults to self.default_provider)
            model: Model name to use (provider-specific)
            system_message: Optional system message for context
            fallback_providers: List of providers to try if the primary fails
            cache: Whether to check/update cache for this request
            **kwargs: Additional parameters to pass to the provider
            
        Returns:
            LLMResponse with the generated text and metadata
        """
        provider = provider or self.default_provider
        fallback_providers = fallback_providers or []
        
        # Determine model if not specified
        if not model:
            if provider == LLMProvider.OPENAI:
                model = "gpt-4"
            elif provider == LLMProvider.ANTHROPIC:
                model = "claude-3-opus-20240229"
            elif provider == LLMProvider.COHERE:
                model = "command"
            elif provider == LLMProvider.MISTRAL:
                model = "mistral-medium"
            elif provider == LLMProvider.GEMINI:
                model = "gemini-pro"
            else:
                raise ValueError(f"No default model specified for provider {provider.value}")
        
        # Update stats
        self.stats["requests"] += 1
        self.stats["provider_calls"][provider.value] += 1
        
        # Check cache
        if cache and self.enable_caching:
            cache_key = self._get_cache_key(provider, model, prompt, kwargs)
            cached_response = self._check_cache(cache_key)
            if cached_response:
                return cached_response
        else:
            cache_key = None
        
        # Generate response
        try:
            if provider == LLMProvider.OPENAI or provider == LLMProvider.OPENAI_SECONDARY:
                response = await self._call_openai(model, prompt, system_message, **kwargs)
            elif provider == LLMProvider.ANTHROPIC:
                response = await self._call_anthropic(model, prompt, system_message, **kwargs)
            elif provider == LLMProvider.COHERE:
                # Implement Cohere API integration
                raise NotImplementedError("Cohere integration not yet implemented")
            elif provider == LLMProvider.MISTRAL:
                # Implement Mistral API integration
                raise NotImplementedError("Mistral integration not yet implemented")
            elif provider == LLMProvider.GEMINI:
                # Implement Gemini API integration  
                raise NotImplementedError("Gemini integration not yet implemented")
            else:
                raise ValueError(f"Unsupported provider: {provider.value}")
            
            # Cache the response
            if cache and self.enable_caching and cache_key:
                self._save_to_cache(cache_key, response)
                
            return response
            
        except Exception as e:
            logger.error(f"Error with provider {provider.value}: {str(e)}")
            
            # Try fallbacks
            for fallback_provider in fallback_providers:
                logger.info(f"Trying fallback provider: {fallback_provider.value}")
                try:
                    return await self.generate(
                        prompt=prompt,
                        provider=fallback_provider,
                        model=None,  # Use default model for fallback
                        system_message=system_message,
                        fallback_providers=[],  # No further fallbacks
                        cache=cache,
                        **kwargs
                    )
                except Exception as fallback_e:
                    logger.error(f"Fallback provider {fallback_provider.value} failed: {str(fallback_e)}")
            
            # If we get here, all providers failed
            raise Exception(f"All LLM providers failed. Original error: {str(e)}")
    
    def generate_sync(
        self, 
        prompt: str,
        provider: Optional[LLMProvider] = None,
        model: Optional[str] = None,
        system_message: Optional[str] = None,
        fallback_providers: Optional[List[LLMProvider]] = None,
        cache: bool = True,
        **kwargs
    ) -> LLMResponse:
        """Synchronous version of generate"""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.generate(
                    prompt=prompt,
                    provider=provider,
                    model=model,
                    system_message=system_message,
                    fallback_providers=fallback_providers,
                    cache=cache,
                    **kwargs
                )
            )
        finally:
            loop.close()
            
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return self.stats
        
    def clear_cache(self):
        """Clear the response cache"""
        if not self.enable_caching:
            return
            
        for filename in os.listdir(self.cache_dir):
            if filename.endswith(".json"):
                os.remove(os.path.join(self.cache_dir, filename))
                
        logger.info("Cache cleared")
