"""LLM Provider factory."""

from typing import Optional, Dict, Type
from .interface import LLMProvider


# Provider registry - lazy loading to avoid import errors
_PROVIDER_CLASSES: Dict[str, Type[LLMProvider]] = {}


def _get_provider_class(provider: str) -> Type[LLMProvider]:
    """Get provider class with lazy loading."""
    if provider not in _PROVIDER_CLASSES:
        if provider == "gemini":
            from .gemini import GeminiProvider
            _PROVIDER_CLASSES["gemini"] = GeminiProvider
        elif provider == "openai":
            from .openai import OpenAIProvider
            _PROVIDER_CLASSES["openai"] = OpenAIProvider
        elif provider == "anthropic":
            from .anthropic import AnthropicProvider
            _PROVIDER_CLASSES["anthropic"] = AnthropicProvider
        elif provider == "ollama":
            from .ollama import OllamaProvider
            _PROVIDER_CLASSES["ollama"] = OllamaProvider
        else:
            raise ValueError(
                f"Unknown provider: {provider}. "
                f"Available providers: gemini, openai, anthropic, ollama"
            )
    return _PROVIDER_CLASSES[provider]


# Default models per provider
DEFAULT_MODELS = {
    "gemini": "gemini-2.0-flash-exp",
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-20250514",
    "ollama": "gemma3:4b"
}

# Default utility models (cheaper/faster for quick evaluations like chime-in)
DEFAULT_UTILITY_MODELS = {
    "gemini": "gemini-2.5-flash-lite",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-20241022",
    "ollama": "gemma3:4b"
}

# Default thinking models (for complex reasoning, architecture decisions)
DEFAULT_THINKING_MODELS = {
    "gemini": "gemini-2.5-pro-exp-03-25",
    "openai": "o1",
    "anthropic": "claude-sonnet-4-20250514",
    "ollama": "qwen2.5:14b"  # Larger model for complex reasoning
}


def create_llm_provider(
    provider: str,
    api_key: str,
    model: Optional[str] = None,
    **kwargs
) -> LLMProvider:
    """
    Factory function to create LLM providers.

    Args:
        provider: Provider name ('gemini', 'openai', 'anthropic')
        api_key: API key for the provider
        model: Model name (uses provider default if not specified)
        **kwargs: Additional provider-specific configuration

    Returns:
        Configured LLMProvider instance

    Raises:
        ValueError: If provider is unknown
        NotImplementedError: If provider is not yet implemented

    Example:
        >>> llm = create_llm_provider(
        ...     provider="gemini",
        ...     api_key="your-api-key",
        ...     model="gemini-2.0-flash-exp"
        ... )
        >>> response = await llm.complete(messages=[...])
    """
    provider_class = _get_provider_class(provider)
    model = model or DEFAULT_MODELS.get(provider)

    return provider_class(api_key=api_key, model=model, **kwargs)


def list_providers() -> list:
    """Return list of available provider names."""
    return list(DEFAULT_MODELS.keys())


def create_utility_llm_from_config(config) -> Optional[LLMProvider]:
    """
    Create a utility LLM provider from agent config settings.

    The utility LLM is used for quick/cheap evaluations like chime-in.
    Falls back to main LLM settings if utility settings not configured.
    Uses DEFAULT_UTILITY_MODELS if no model specified.

    Args:
        config: Agent config with utility_llm_provider, utility_llm_api_key,
                utility_llm_model, and main LLM settings for fallback

    Returns:
        LLMProvider instance for utility tasks, or None if no LLM configured
    """
    # Determine provider (utility -> main fallback)
    provider = getattr(config, 'utility_llm_provider', None) or getattr(config, 'llm_provider', None)
    if not provider:
        return None

    # Determine API key (utility -> main fallback)
    api_key = getattr(config, 'utility_llm_api_key', None) or getattr(config, 'llm_api_key', None)
    if not api_key:
        return None

    # Determine model (utility -> default utility model for provider)
    model = getattr(config, 'utility_llm_model', None)
    if not model:
        model = DEFAULT_UTILITY_MODELS.get(provider)

    try:
        kwargs = {}
        # Pass Ollama base URL if configured
        if provider == "ollama":
            base_url = getattr(config, 'ollama_base_url', None)
            if base_url:
                kwargs["base_url"] = base_url

        return create_llm_provider(provider=provider, api_key=api_key, model=model, **kwargs)
    except Exception:
        return None


def create_thinking_llm_from_config(config) -> Optional[LLMProvider]:
    """
    Create a thinking LLM provider from agent config settings.

    The thinking LLM is used for complex reasoning, architecture decisions,
    and tasks that benefit from deeper analysis.
    Falls back to main LLM settings if thinking settings not configured.
    Uses DEFAULT_THINKING_MODELS if no model specified.

    Args:
        config: Agent config with thinking_llm_provider, thinking_llm_api_key,
                thinking_llm_model, and main LLM settings for fallback

    Returns:
        LLMProvider instance for thinking tasks, or None if no LLM configured
    """
    # Determine provider (thinking -> main fallback)
    provider = getattr(config, 'thinking_llm_provider', None) or getattr(config, 'llm_provider', None)
    if not provider:
        return None

    # Determine API key (thinking -> main fallback)
    api_key = getattr(config, 'thinking_llm_api_key', None) or getattr(config, 'llm_api_key', None)
    if not api_key:
        return None

    # Determine model (thinking -> default thinking model for provider)
    model = getattr(config, 'thinking_llm_model', None)
    if not model:
        model = DEFAULT_THINKING_MODELS.get(provider)

    try:
        kwargs = {}
        # Pass Ollama base URL if configured
        if provider == "ollama":
            base_url = getattr(config, 'ollama_base_url', None)
            if base_url:
                kwargs["base_url"] = base_url

        return create_llm_provider(provider=provider, api_key=api_key, model=model, **kwargs)
    except Exception:
        return None
