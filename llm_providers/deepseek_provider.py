from typing import Dict, Any
from openai import OpenAI
from .base import BaseLLMProvider
from utils.network_checker import network_checker

class DeepSeekProvider(BaseLLMProvider):
    """DeepSeek API provider implementation using OpenAI SDK."""
    
    def __init__(self, api_key: str, enable_web_search: bool = True, **kwargs):
        super().__init__(api_key, enable_web_search, **kwargs)
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        
        # Check web search availability at initialization
        # Note: DeepSeek doesn't have web search yet, but we're future-proofing
        if self.enable_web_search:
            self._web_search_available = network_checker.is_web_search_available('deepseek')
            if not self._web_search_available:
                print("⚠️ Web search may not be available for DeepSeek due to connectivity issues")
                print(network_checker.get_connectivity_report('deepseek'))
        else:
            self._web_search_available = False
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using DeepSeek's API via OpenAI SDK."""
        try:
            # Note: DeepSeek doesn't have web search tools yet, but we're ready for when they do
            if self.enable_web_search and self._web_search_available:
                print("🔍 Web search requested but not yet supported by DeepSeek")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"DeepSeek API error: {e}")
    
    def get_name(self) -> str:
        return f"DeepSeek ({self.model})"
    
    def get_available_models(self) -> Dict[str, Any]:
        """Returns a dictionary of available models for DeepSeek."""
        return {
            "deepseek-chat": {"max_tokens": 4096, "description": "General-purpose chat model"},
            "deepseek-coder": {"max_tokens": 16384, "description": "Code generation model"},
            "deepseek-reasoner": {"max_tokens": 16384, "description": "Advanced reasoning model"},
        }
    
    @property
    def supports_pdf_upload(self) -> bool:
        """Indicates that DeepSeek provider does not support direct PDF uploads."""
        return False
