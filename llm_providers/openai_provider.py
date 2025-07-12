from typing import Dict, Any, Optional, Union, List
import base64
from openai import OpenAI
from .base import BaseLLMProvider
from utils.network_checker import network_checker

class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider implementation with GPT-4o and native PDF support."""
    
    def __init__(self, api_key: str, enable_web_search: bool = True, **kwargs):
        super().__init__(api_key, enable_web_search, **kwargs)
        self.client = OpenAI(api_key=api_key)
        self._file_id_cache = {}  # Cache for uploaded file IDs
        
        # Check web search availability at initialization
        # Note: OpenAI doesn't have web search yet, but we're future-proofing
        if self.enable_web_search:
            self._web_search_available = network_checker.is_web_search_available('openai')
            if not self._web_search_available:
                print("⚠️ Web search may not be available for OpenAI due to connectivity issues")
                print(network_checker.get_connectivity_report('openai'))
        else:
            self._web_search_available = False
    
    def generate(self, prompt: Union[str, List[str]], pdf_path: Optional[str] = None) -> str:
        """Generate text using OpenAI's API with optional PDF support."""
        if isinstance(prompt, list):
            prompt_text = "\n".join(str(p) for p in prompt)
        else:
            prompt_text = prompt

        api_input = [{"role": "user", "content": []}]

        if pdf_path:
            # Check if file is already cached
            if pdf_path in self._file_id_cache:
                print(f"📋 Using cached file ID for {pdf_path}")
                file_id = self._file_id_cache[pdf_path]
            else:
                # Upload the file
                print(f"📤 Uploading PDF file: {pdf_path}")
                try:
                    with open(pdf_path, "rb") as f:
                        uploaded_file = self.client.files.create(file=f, purpose="user_data")
                    file_id = uploaded_file.id
                    # Cache the file ID
                    self._file_id_cache[pdf_path] = file_id
                    print(f"✅ PDF uploaded and cached with ID: {file_id}")
                except Exception as e:
                    print(f"❌ Failed to upload PDF: {e}")
                    raise Exception(f"OpenAI file upload error: {e}")
            
            # Add file and text to content
            api_input[0]["content"].append({"type": "input_file", "file_id": file_id})
            api_input[0]["content"].append({"type": "input_text", "text": prompt_text})
        else:
            # Text-only request
            api_input[0]["content"].append({"type": "input_text", "text": prompt_text})

        try:
            # Note: OpenAI doesn't have web search tools yet, but we're ready for when they do
            if self.enable_web_search and self._web_search_available:
                print("🔍 Web search requested but not yet supported by OpenAI")
            
            response = self.client.responses.create(
                model=self.model,
                input=api_input
            )
            
            return response.output_text
            
        except Exception as e:
            raise Exception(f"OpenAI API error: {e}")
    
    def get_name(self) -> str:
        return "OpenAI (GPT-4o)"
    
    def clear_file_cache(self):
        """Clear cached file IDs."""
        self._file_id_cache.clear()
        print("🗑️ Cleared OpenAI file cache")
    
    def get_available_models(self) -> Dict[str, Any]:
        """Returns a dictionary of available models for OpenAI."""
        return {
            "gpt-4o": {"max_tokens": 16384, "description": "Most capable GPT-4 model with vision"},
            "gpt-4o-mini": {"max_tokens": 16384, "description": "Fast and efficient GPT-4 model"},
            "o3-mini": {"max_tokens": 65536, "description": "Advanced reasoning model"},
        }
    
    @property
    def supports_pdf_upload(self) -> bool:
        """Indicates that OpenAI provider supports direct PDF uploads."""
        return True
