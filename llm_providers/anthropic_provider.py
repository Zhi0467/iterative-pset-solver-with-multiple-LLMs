from typing import Dict, Any, Optional, Union, List
import base64
import os
from anthropic import Anthropic
from .base import BaseLLMProvider
from utils.network_checker import network_checker

class AnthropicProvider(BaseLLMProvider):
    """Anthropic API provider implementation with Files API for PDFs and web search."""
    
    def __init__(self, api_key: str, enable_web_search: bool = True, **kwargs):
        super().__init__(api_key, enable_web_search, **kwargs)
        
        # Handle custom base URL and proxy mode
        self.base_url = kwargs.get('base_url')
        self.proxy_mode = kwargs.get('proxy_mode', False)
        self.system = kwargs.get('system')
        
        if self.base_url:
            self.client = Anthropic(api_key=api_key, base_url=self.base_url)
            # self.proxy_mode = True
            print(f"🔗 Using custom Anthropic base URL: {self.base_url}")
        else:
            self.client = Anthropic(api_key=api_key)
        
        # Cache for file IDs to avoid re-uploading
        self._file_id_cache = {}
        
        # Check web search availability at initialization
        if self.enable_web_search:
            self._web_search_available = network_checker.is_web_search_available('anthropic')
            if not self._web_search_available:
                print("⚠️ Web search may not be available for Anthropic due to connectivity issues")
                print(network_checker.get_connectivity_report('anthropic'))
        else:
            self._web_search_available = False
    
    def generate(self, 
                prompt: Union[str, List[str]], 
                pdf_path: Optional[str] = None, 
                pdf_paths: Optional[List[str]] = None) -> str:
        """Generate text using Anthropic's API with Files API for PDFs and web search.
        
        Args:
            prompt: String or list of strings containing the prompt(s)
            pdf_path: Optional path to PDF file to include
            pdf_paths: Optional list of paths to PDF files to include (backward compatibility)
        """
        # Convert prompt to list if it's a string
        prompts = [prompt] if isinstance(prompt, str) else prompt
        
        # Build message content
        content = []
        
        # Backward-compat: if pdf_path not given but pdf_paths list is, use its first element
        if pdf_path is None and pdf_paths:
            pdf_path = pdf_paths[0]
        
        # Handle PDF content
        if pdf_path:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            if self.proxy_mode:
                # For proxy mode, encode PDF as base64
                print(f"📖 Reading PDF for base64 encoding: {pdf_path}")
                with open(pdf_path, 'rb') as f:
                    pdf_data = f.read()
                    pdf_base64 = base64.b64encode(pdf_data).decode('utf-8')
                
                content.append({
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": pdf_base64
                    }
                })
            else:
                # For direct mode, use Files API
                if pdf_path in self._file_id_cache:
                    print(f"📋 Using cached file ID for {pdf_path}")
                    file_id = self._file_id_cache[pdf_path]
                else:
                    print(f"📤 Uploading PDF file: {pdf_path}")
                    with open(pdf_path, 'rb') as f:
                        uploaded_file = self.client.beta.files.upload(
                            file=(os.path.basename(pdf_path), f, "application/pdf")
                        )
                    file_id = uploaded_file.id
                    self._file_id_cache[pdf_path] = file_id
                    print(f"✅ PDF uploaded and cached with ID: {file_id}")
                
                content.append({
                    "type": "document",
                    "source": {
                        "type": "file",
                        "file_id": file_id
                    }
                })
        
        # Add text prompts
        for prompt_text in prompts:
            content.append({
                "type": "text",
                "text": str(prompt_text)
            })
        
        # Prepare message creation parameters
        message_params = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": content}]
        }
        
        # Add system message if provided
        if self.system:
            message_params["system"] = self.system
        
        # Try with web search first, then fallback if needed
        web_search_attempted = False
        if self.enable_web_search and self._web_search_available:
            try:
                message_params["tools"] = [{
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": 5
                }]

                web_search_attempted = True
                
                # For non-proxy mode with PDFs, use the beta API
                if pdf_path and not self.proxy_mode:
                    message_params["betas"] = ["files-api-2025-04-14"]
                    message = self.client.beta.messages.create(**message_params)
                else:
                    # For proxy mode or text-only, use the standard API
                    message = self.client.messages.create(**message_params)
                
                return self._extract_text_from_response(message)
                
            except Exception as e:
                error_str = str(e).lower()
                if any(keyword in error_str for keyword in ['connect', 'timeout', 'network', 'dns']):
                    print(f"⚠️ Web search failed due to connectivity issue: {e}")
                    print("🔄 Retrying without web search...")
                    # Remove web search tools and continue to fallback
                    message_params.pop("tools", None)
                else:
                    # Re-raise non-connectivity errors
                    raise
        
        # Fallback: Generate without web search
        if web_search_attempted:
            print("📝 Generating response without web search")
        elif self.enable_web_search and not self._web_search_available:
            print("📝 Generating response without web search (connectivity unavailable)")
        
        # For non-proxy mode with PDFs, use the beta API
        if pdf_path and not self.proxy_mode:
            message_params["betas"] = ["files-api-2025-04-14"]
            message = self.client.beta.messages.create(**message_params)
        else:
            # For proxy mode or text-only, use the standard API
            message = self.client.messages.create(**message_params)
        
        # Extract and return the text content from the response
        return self._extract_text_from_response(message)
    
    def _extract_text_from_response(self, message) -> str:
        """Extract text content from Claude's response, handling web search results."""
        text_parts = []
        
        for content_block in message.content:
            if hasattr(content_block, 'type'):
                if content_block.type == 'text':
                    text_parts.append(content_block.text)
                elif content_block.type == 'server_tool_use':
                    # Log web search queries for debugging
                    if content_block.name == 'web_search':
                        query = content_block.input.get('query', 'unknown')
                        print(f"🔍 Claude searched for: '{query}'")
                elif content_block.type == 'web_search_tool_result':
                    # Log search results for debugging
                    results_count = len(content_block.content) if hasattr(content_block, 'content') else 0
                    print(f"📋 Found {results_count} web search results")
            else:
                # Fallback for simple text content
                if hasattr(content_block, 'text'):
                    text_parts.append(content_block.text)
        
        return ''.join(text_parts)
    
    def get_name(self) -> str:
        return "Anthropic (Claude 4)"
    
    def clear_file_cache(self):
        """Clear cached file IDs."""
        self._file_id_cache.clear()
        print("🗑️ Cleared Anthropic file cache")
    
    def get_available_models(self) -> Dict[str, Any]:
        """Returns a dictionary of available models for Anthropic."""
        return {
            "claude-3-5-sonnet-20241022": {"max_tokens": 8192, "description": "Most capable Claude model"},
            "claude-3-5-haiku-20241022": {"max_tokens": 8192, "description": "Fast and efficient Claude model"},
            "claude-sonnet-4-20250514": {"max_tokens": 16384, "description": "Latest Claude 4 model"},
        }
    
    @property
    def supports_pdf_upload(self) -> bool:
        """Indicates that Anthropic provider supports direct PDF uploads."""
        return True
