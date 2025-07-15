from typing import Dict, Any, Optional, Union, List
from google import genai
from google.genai import types
import pathlib
import os
from .base import BaseLLMProvider
from utils.network_checker import network_checker

class GeminiProvider(BaseLLMProvider):
    """Google Gemini API provider implementation using google.genai package."""

    def __init__(self, api_key: str, enable_web_search: bool = True, **kwargs):
        super().__init__(api_key, enable_web_search, **kwargs)
        # Initialize the client with API key
        self.client = genai.Client(api_key=api_key)
        self._file_content_cache = {}  # Cache for PDF file content
        
        # Check web search availability at initialization
        if self.enable_web_search:
            self._web_search_available = network_checker.is_web_search_available('gemini')
            if not self._web_search_available:
                print("⚠️ Web search may not be available for Gemini due to connectivity issues")
                print(network_checker.get_connectivity_report('gemini'))
        else:
            self._web_search_available = False

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def generate(
        self,
        prompt: Union[str, List[str]],
        pdf_paths: Optional[List[str]] = None,
        pdf_path: Optional[str] = None,
    ) -> str:
        """Generate text using Google's Gemini API with optional Google Search grounding.

        Args:
            prompt: A single string or list of prompt strings.
            pdf_paths: Optional list of PDF file paths (for backward compatibility).
            pdf_path: Optional path to a single PDF file.
        """
        # Backward-compat: if pdf_path not given but pdf_paths provided, take first element
        if pdf_path is None and pdf_paths:
            pdf_path = pdf_paths[0]

        # Prepare contents list
        contents = []

        # Add PDF part first (if any)
        if pdf_path:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            # Check if file content is already cached
            if pdf_path in self._file_content_cache:
                print(f"📋 Using cached PDF content for {pdf_path}")
                pdf_data = self._file_content_cache[pdf_path]
            else:
                print(f"📖 Reading PDF file: {pdf_path}")
                filepath = pathlib.Path(pdf_path)
                pdf_data = filepath.read_bytes()
                # Cache the file content
                self._file_content_cache[pdf_path] = pdf_data
                print(f"✅ PDF content cached for {pdf_path}")
            
            pdf_part = types.Part.from_bytes(
                data=pdf_data,
                mime_type='application/pdf',
            )
            contents.append(pdf_part)

        # Add prompt text
        if isinstance(prompt, list):
            contents.extend(prompt)
        else:
            contents.append(prompt)

        # Prepare config with optional Google Search grounding
        config_params = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
        }
        
        # Try with web search first, then fallback if needed
        web_search_attempted = False
        if self.enable_web_search and self._web_search_available:
            try:
                grounding_tool = types.Tool(google_search=types.GoogleSearch())
                config_params["tools"] = [grounding_tool]
                web_search_attempted = True
                
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=types.GenerateContentConfig(**config_params)
                )
                
                return self._extract_text_from_response(response)
                
            except Exception as e:
                error_str = str(e).lower()
                if any(keyword in error_str for keyword in ['connect', 'timeout', 'network', 'dns', 'nodename', 'servname']):
                    print(f"⚠️ Web search failed due to connectivity issue: {e}")
                    print("🔄 Retrying without web search...")
                    # Remove web search tools and continue to fallback
                    config_params.pop("tools", None)
                else:
                    # Re-raise non-connectivity errors
                    raise
        
        # Fallback: Generate without web search
        if web_search_attempted:
            print("📝 Generating response without web search")
        elif self.enable_web_search and not self._web_search_available:
            print("📝 Generating response without web search (connectivity unavailable)")
        
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(**config_params)
            )
            
            return self._extract_text_from_response(response)
            
        except Exception as e:
            raise Exception(f"Gemini API error: {e}")

    def _extract_text_from_response(self, response) -> str:
        """Extract text content from Gemini's response - simplified approach."""
        try:
            return response.text
        except Exception as e:
            raise Exception(f"Gemini API: Could not extract text from response. Error: {e}")

    # ---------------------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------------------
    def get_name(self) -> str:
        return f"Google ({self.model})"
    
    def clear_file_cache(self):
        """Clear cached file content."""
        self._file_content_cache.clear()
        print("🗑️ Cleared Gemini file cache")

    def get_available_models(self) -> Dict[str, Any]:
        return {
            "gemini-2.5-flash": {
                "description": "Fast and efficient model for most tasks",
                "max_tokens": 8192
            },
            "gemini-2.5-pro": {
                "description": "Most capable Gemini model for complex tasks",
                "max_tokens": 8192
            },
            "gemini-pro": {
                "description": "Well-rounded model for a variety of tasks",
                "max_tokens": 8192
            }
        }
