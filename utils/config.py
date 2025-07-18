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
    """
    for simple tasks, use:
    - claude-3-5-sonnet-20241022, max_tokens: 8192
    - claude-3-5-haiku-20241022, max_tokens: 8192
    - gemini-2.5-flash, max_tokens: 32768
    - deepseek-chat, max_tokens: 8192

    for complex tasks, use:
    - claude-sonnet-4-20250514, max_tokens: 16384
    - claude-opus-4-20250514, max_tokens: 16384
    - gemini-2.5-pro, max_tokens: 32768
    - deepseek-reasoner, max_tokens: 16384

    while gpt-4o is good for simple tasks and avoid using it for complex tasks.
    """
    
    # Default web search settings for each provider
    DEFAULT_WEB_SEARCH = {
        PROVIDER_OPENAI: False,      # OpenAI doesn't support web search yet
        PROVIDER_ANTHROPIC: True,    # Anthropic supports web search
        PROVIDER_GEMINI: False,       # Gemini supports Google Search grounding
        PROVIDER_DEEPSEEK: False     # DeepSeek doesn't support web search yet
    }
    
    # Default code execution settings for each provider
    DEFAULT_CODE_EXECUTION = {
        PROVIDER_OPENAI: False,      # OpenAI doesn't support direct code execution
        PROVIDER_ANTHROPIC: True,    # Anthropic supports direct code execution
        PROVIDER_GEMINI: False,      # Gemini doesn't support direct code execution
        PROVIDER_DEEPSEEK: False     # DeepSeek doesn't support direct code execution
    }
    
    # Default MCP settings for each provider
    DEFAULT_MCP = {
        PROVIDER_OPENAI: False,      # OpenAI doesn't support MCP integration yet
        PROVIDER_ANTHROPIC: False,    # Anthropic can integrate with MCP tools
        PROVIDER_GEMINI: False,      # Gemini doesn't support MCP integration yet
        PROVIDER_DEEPSEEK: False     # DeepSeek doesn't support MCP integration yet
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
                "temperature": 0.3,
                "max_tokens": 32768,
                "enable_web_search": self.DEFAULT_WEB_SEARCH[self.PROVIDER_OPENAI],
                "enable_code_execution": self.DEFAULT_CODE_EXECUTION[self.PROVIDER_OPENAI],
                "enable_mcp": self.DEFAULT_MCP[self.PROVIDER_OPENAI],
            },
            self.PROVIDER_ANTHROPIC: {
                "model": self.DEFAULT_MODELS[self.PROVIDER_ANTHROPIC],
                "temperature": 0.1,
                "max_tokens": 16384,
                "enable_web_search": self.DEFAULT_WEB_SEARCH[self.PROVIDER_ANTHROPIC],
                "enable_code_execution": self.DEFAULT_CODE_EXECUTION[self.PROVIDER_ANTHROPIC],
                "enable_mcp": self.DEFAULT_MCP[self.PROVIDER_ANTHROPIC],
                "mcp_server_url": "stdio",
                "mcp_server_command": ["python", "mcp_server/math_tools.py"]
            },
            self.PROVIDER_GEMINI: {
                "model": self.DEFAULT_MODELS[self.PROVIDER_GEMINI],
                "temperature": 0.1,
                "max_tokens": 32768,
                "enable_web_search": self.DEFAULT_WEB_SEARCH[self.PROVIDER_GEMINI],
                "enable_code_execution": self.DEFAULT_CODE_EXECUTION[self.PROVIDER_GEMINI],
                "enable_mcp": self.DEFAULT_MCP[self.PROVIDER_GEMINI],
            },
            self.PROVIDER_DEEPSEEK: {
                "model": self.DEFAULT_MODELS[self.PROVIDER_DEEPSEEK],
                "temperature": 0.0,
                "max_tokens": 16384,
                "enable_web_search": self.DEFAULT_WEB_SEARCH[self.PROVIDER_DEEPSEEK],
                "enable_code_execution": self.DEFAULT_CODE_EXECUTION[self.PROVIDER_DEEPSEEK],
                "enable_mcp": self.DEFAULT_MCP[self.PROVIDER_DEEPSEEK],
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

    # Backward compatibility alias used in tests
    def get_llm_config(self, provider: str) -> Dict[str, Any]:
        """Alias for get_provider_config to maintain legacy API used in tests."""
        return self.get_provider_config(provider)

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
    
    def set_code_execution_enabled(self, provider: str, enabled: bool):
        """Enable or disable code execution for a specific provider."""
        if provider in self.provider_configs:
            self.provider_configs[provider]["enable_code_execution"] = enabled
            print(f"🔧 Code execution {'enabled' if enabled else 'disabled'} for {provider}")

    def get_code_execution_enabled(self, provider: str) -> bool:
        """Check if code execution is enabled for a provider."""
        return self.provider_configs.get(provider, {}).get("enable_code_execution", False)
    
    def get_code_execution_status(self) -> Dict[str, bool]:
        """Get code execution status for all providers."""
        return {
            provider: self.get_code_execution_enabled(provider)
            for provider in [
                self.PROVIDER_OPENAI,
                self.PROVIDER_ANTHROPIC,
                self.PROVIDER_GEMINI,
                self.PROVIDER_DEEPSEEK
            ]
        }
    
    def set_mcp_enabled(self, provider: str, enabled: bool):
        """Enable or disable MCP integration for a specific provider."""
        if provider in self.provider_configs:
            self.provider_configs[provider]["enable_mcp"] = enabled
            print(f"🧮 MCP integration {'enabled' if enabled else 'disabled'} for {provider}")

    def get_mcp_enabled(self, provider: str) -> bool:
        """Check if MCP integration is enabled for a provider."""
        return self.provider_configs.get(provider, {}).get("enable_mcp", False)
    
    def get_mcp_status(self) -> Dict[str, bool]:
        """Get MCP status for all providers."""
        return {
            provider: self.get_mcp_enabled(provider)
            for provider in [
                self.PROVIDER_OPENAI,
                self.PROVIDER_ANTHROPIC,
                self.PROVIDER_GEMINI,
                self.PROVIDER_DEEPSEEK
            ]
        }
    
    def set_mcp_server_config(self, provider: str, server_url: str, server_command: Optional[list] = None):
        """Set MCP server configuration for a provider."""
        if provider in self.provider_configs:
            self.provider_configs[provider]["mcp_server_url"] = server_url
            if server_command:
                self.provider_configs[provider]["mcp_server_command"] = server_command
            print(f"🧮 MCP server configuration updated for {provider}")
    
    def get_provider_capabilities(self, provider: str) -> Dict[str, bool]:
        """Get all capabilities for a specific provider."""
        config = self.provider_configs.get(provider, {})
        return {
            "web_search": config.get("enable_web_search", False),
            "code_execution": config.get("enable_code_execution", False),
            "mcp": config.get("enable_mcp", False),
            "pdf_upload": True  # Most providers support PDF upload
        }
