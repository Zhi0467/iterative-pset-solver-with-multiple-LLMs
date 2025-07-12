import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LLMConfig:
    """Configuration for LLM providers."""
    
    # Available provider types
    PROVIDER_OPENAI = "openai"
    PROVIDER_ANTHROPIC = "anthropic"
    PROVIDER_GEMINI = "gemini"
    PROVIDER_DEEPSEEK = "deepseek"
    
    # Base URLs for providers that need it
    BASE_URLS = {
        PROVIDER_ANTHROPIC: os.getenv("ANTHROPIC_BASE_URL")
    }
    
    # Default models for each provider
    DEFAULT_MODELS = {
        PROVIDER_OPENAI: "gpt-4o",
        PROVIDER_ANTHROPIC: "claude-sonnet-4-20250514",
        PROVIDER_GEMINI: "gemini-2.5-pro",
        PROVIDER_DEEPSEEK: "deepseek-reasoner"
    }
    
    # Default web search settings for each provider
    DEFAULT_WEB_SEARCH = {
        PROVIDER_OPENAI: False,      # OpenAI doesn't support web search yet
        PROVIDER_ANTHROPIC: True,    # Anthropic supports web search
        PROVIDER_GEMINI: True,       # Gemini supports Google Search grounding
        PROVIDER_DEEPSEEK: False     # DeepSeek doesn't support web search yet
    }
    
    def __init__(self):
        # Load API keys from environment
        self.api_keys = {
            self.PROVIDER_OPENAI: os.getenv("OPENAI_API_KEY"),
            self.PROVIDER_ANTHROPIC: os.getenv("ANTHROPIC_API_KEY"),
            self.PROVIDER_GEMINI: os.getenv("GOOGLE_API_KEY"),
            self.PROVIDER_DEEPSEEK: os.getenv("DEEPSEEK_API_KEY")
        }
        
        # Default provider configurations
        self.provider_configs = {
            self.PROVIDER_OPENAI: {
                "model": self.DEFAULT_MODELS[self.PROVIDER_OPENAI],
                "temperature": 0.1,
                "max_tokens": 16384,
                "enable_web_search": self.DEFAULT_WEB_SEARCH[self.PROVIDER_OPENAI]
            },
            self.PROVIDER_ANTHROPIC: {
                "model": self.DEFAULT_MODELS[self.PROVIDER_ANTHROPIC],
                "temperature": 0.1,
                "max_tokens": 16384,
                "enable_web_search": self.DEFAULT_WEB_SEARCH[self.PROVIDER_ANTHROPIC]
            },
            self.PROVIDER_GEMINI: {
                "model": self.DEFAULT_MODELS[self.PROVIDER_GEMINI],
                "temperature": 0.1,
                "max_tokens": 32768,
                "enable_web_search": self.DEFAULT_WEB_SEARCH[self.PROVIDER_GEMINI]
            },
            self.PROVIDER_DEEPSEEK: {
                "model": self.DEFAULT_MODELS[self.PROVIDER_DEEPSEEK],
                "temperature": 0.0,
                "max_tokens": 16384,
                "enable_web_search": self.DEFAULT_WEB_SEARCH[self.PROVIDER_DEEPSEEK]
            }
        }
        
        # Base URLs for different providers
        self.base_urls = {
            self.PROVIDER_OPENAI: "https://api.openai.com/v1",
            self.PROVIDER_ANTHROPIC: "https://api.anthropic.com",
            self.PROVIDER_GEMINI: "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent?key=" + self.api_keys[self.PROVIDER_GEMINI],
            self.PROVIDER_DEEPSEEK: "https://api.deepseek.com/v1"
        }

    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for a provider."""
        return self.api_keys.get(provider)

    def get_base_url(self, provider: str) -> Optional[str]:
        """Get base URL for a provider."""
        return self.BASE_URLS.get(provider)

    def get_provider_config(self, provider: str) -> Dict[str, Any]:
        """Get configuration for a provider."""
        return self.provider_configs.get(provider, {}).copy()

    def set_provider_config(self, provider: str, config: Dict[str, Any]):
        """Update configuration for a provider."""
        if provider in self.provider_configs:
            self.provider_configs[provider].update(config)

    def set_web_search_enabled(self, provider: str, enabled: bool):
        """Enable or disable web search for a specific provider."""
        if provider in self.provider_configs:
            self.provider_configs[provider]["enable_web_search"] = enabled
            print(f"🔍 Web search {'enabled' if enabled else 'disabled'} for {provider}")

    def get_web_search_enabled(self, provider: str) -> bool:
        """Check if web search is enabled for a provider."""
        return self.provider_configs.get(provider, {}).get("enable_web_search", False)

    def is_provider_available(self, provider: str) -> bool:
        """Check if a provider is available (has API key)."""
        return bool(self.get_api_key(provider))

    def get_available_providers(self) -> Dict[str, bool]:
        """Get all providers and their availability status."""
        return {
            provider: self.is_provider_available(provider)
            for provider in [
                self.PROVIDER_OPENAI,
                self.PROVIDER_ANTHROPIC,
                self.PROVIDER_GEMINI,
                self.PROVIDER_DEEPSEEK
            ]
        }
    
    def get_web_search_status(self) -> Dict[str, bool]:
        """Get web search status for all providers."""
        return {
            provider: self.get_web_search_enabled(provider)
            for provider in [
                self.PROVIDER_OPENAI,
                self.PROVIDER_ANTHROPIC,
                self.PROVIDER_GEMINI,
                self.PROVIDER_DEEPSEEK
            ]
        }
