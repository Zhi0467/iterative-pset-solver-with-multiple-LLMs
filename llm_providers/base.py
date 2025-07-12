from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseLLMProvider(ABC):
    """Base class for LLM providers."""
    
    def __init__(self, api_key: str, enable_web_search: bool = True, **kwargs):
        self.api_key = api_key
        self.model = kwargs.get("model")
        self.temperature = kwargs.get("temperature")
        self.max_tokens = kwargs.get("max_tokens")
        self.enable_web_search = enable_web_search
        self.kwargs = kwargs

    @abstractmethod
    def generate(self, prompt, **kwargs) -> str:
        """Generate text from the LLM.
        
        Args:
            prompt: The input prompt (str or List[str])
            **kwargs: Additional parameters like pdf_path, etc.
            
        Returns:
            Generated text response
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get the name of the LLM provider."""
        pass

    @abstractmethod
    def get_available_models(self) -> Dict[str, Any]:
        """Get a list of available models for the provider."""
        pass

    @property
    def supports_pdf_upload(self) -> bool:
        """Indicates if the provider supports direct PDF uploads."""
        # By default, providers are assumed to support it unless they override this.
        return True
    
    def clear_file_cache(self):
        """Clear any cached file data. Override in subclasses if needed."""
        pass
