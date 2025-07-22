from typing import Dict, Any, Optional, Union, List
import base64
import os
import logging
import re
from anthropic import Anthropic
from .base import BaseLLMProvider
from utils.network_checker import network_checker
from utils.sandbox import create_sandbox

# Try to import MCP client, but don't fail if not available
try:
    from mcp_server.client import SyncMCPClient, get_mcp_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    SyncMCPClient = None
    get_mcp_client = None

logger = logging.getLogger(__name__)

MAX_TOOL_TURNS = 8  # Hard cap on recursive tool turns per conversation

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
        
        # Initialize MCP client if enabled and available (isolated instance)
        self._mcp_client = None
        if self.enable_mcp and MCP_AVAILABLE and get_mcp_client:
            try:
                mcp_server_command = kwargs.get('mcp_server_command', 
                    ['python', os.path.join(os.path.dirname(__file__), '..', 'mcp_server', 'math_tools.py')])
                self._mcp_client = get_mcp_client(mcp_server_command)
                if self._mcp_client:
                    print(f"🧮 Connected to MCP server with {len(self._mcp_client.get_available_tools())} math tools")
                else:
                    print("⚠️ Failed to connect to MCP server")
            except Exception as e:
                print(f"⚠️ MCP initialization failed: {e}")
                self._mcp_client = None
        elif self.enable_mcp and not MCP_AVAILABLE:
            print("⚠️ MCP requested but dependencies not available. Install with: pip install -r mcp_requirements.txt")
        
        # Initialize code execution sandbox
        self._sandbox = None
        if self.enable_code_execution:
            try:
                use_docker = kwargs.get('use_docker_sandbox', False)
                self._sandbox = create_sandbox(
                    use_docker=use_docker,
                    timeout=kwargs.get('execution_timeout', 30),
                    max_memory_mb=kwargs.get('max_memory_mb', 512),
                    enable_network=kwargs.get('enable_network', False)
                )
                print(f"🔧 Code execution sandbox initialized (Docker: {use_docker})")
            except Exception as e:
                print(f"⚠️ Sandbox initialization failed: {e}")
                self._sandbox = None
    
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
                    },
                    "cache_control": {"type": "ephemeral"}
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
                    },
                    "cache_control": {"type": "ephemeral"}
                })
        
        # Add text prompts
        for prompt_text in prompts:
            text_content = {
                "type": "text",
                "text": str(prompt_text)
            }
            # Cache larger prompts (>500 chars) that are likely to be reused
            if len(str(prompt_text)) > 500:
                text_content["cache_control"] = {"type": "ephemeral"}
            content.append(text_content)
        
        # Prepare message creation parameters
        message_params = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": content}]
        }
        
        # Add system message if provided with cache control for prompt caching
        if self.system:
            # System messages are ideal for caching as they're stable and reusable
            message_params["system"] = [
                {
                    "type": "text",
                    "text": self.system,
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        
        # Add tools (web search, code execution, MCP tools)
        tools = []
        
        # Add web search tool
        if self.enable_web_search and self._web_search_available:
            tools.append({
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 5
            })
        
        # Add code execution tool
        if self.enable_code_execution:
            tools.append({
                "type": "bash_20250124",
                "name": "bash"
            })
            print(f"🖥️ [BASH] Bash tool registered for code execution")
        
        # Add MCP tools - no longer registering evaluate_expression directly
        # Will be handled through mcp_generic_math with fallback logic
        
        # Add a single versatile math tool for difficult numerical calculations
        if self._mcp_client:
            versatile_tool_schema = {
                "name": "mcp_generic_math",
                "description": "Interface to call advanced numerical MCP math tools for difficult calculations",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "tool": {
                            "type": "string",
                            "enum": ["solve_equation", "differentiate", "integrate", "definite_integral", "matrix_multiply", 
                                    "matrix_determinant", "matrix_inverse", "linear_regression", "factorial", "combination", "permutation"],
                            "description": "Name of the advanced numerical MCP math tool to invoke for difficult calculations."
                        },
                        "args": {
                            "type": "object",
                            "description": "Arguments object to pass directly to the chosen MCP tool."
                        }
                    },
                    "required": ["tool", "args"]
                }
            }
            tools.append(versatile_tool_schema)
        
        # Try with tools first, then fallback if needed
        web_search_attempted = False
        if tools:
            try:
                # Add cache control to tools for prompt caching - tool definitions are stable and reusable
                for tool in tools:
                    tool["cache_control"] = {"type": "ephemeral"}
                message_params["tools"] = tools
                if logger.isEnabledFor(logging.DEBUG):
                    print(f"🔧 Registered {len(tools)} tools: {[tool.get('name', tool.get('type', 'unknown')) for tool in tools]}")
                else:
                    print(f"🔧 Registered {len(tools)} tools for {self.get_name()}")
                web_search_attempted = self.enable_web_search and self._web_search_available
                
                message_params["stream"] = False  # default non-streaming
                # For non-proxy mode with PDFs, use the beta API
                if pdf_path and not self.proxy_mode:
                    message_params["betas"] = ["files-api-2025-04-14"]
                    message = self.client.beta.messages.create(**message_params)
                else:
                    # For proxy mode or text-only, use the standard API
                    message = self.client.messages.create(**message_params)

                # Handle tool use conversation if needed
                return self._handle_tool_use_conversation(message, message_params, pdf_path, tool_turn_count=0)
            except Exception as e:
                # Some newer Anthropic SDK versions raise an exception for long / large
                # requests and recommend using streaming mode instead. In that case we
                # transparently retry the same request with streaming enabled and
                # collect the streamed content before returning it so that the rest
                # of the pipeline can keep working without changes.
                err_msg = str(e).lower()
                if "streaming is strongly recommended" in err_msg or "with_streaming_response" in err_msg or "long-requests" in err_msg:
                    print("⚠️ Large request detected – retrying with streaming mode enabled ...")
                    # Enable streaming mode and retry
                    message_params["stream"] = True
                    try:
                        if pdf_path and not self.proxy_mode:
                            message_params["betas"] = ["files-api-2025-04-14"]
                            stream = self.client.beta.messages.create(**message_params)
                        else:
                            stream = self.client.messages.create(**message_params)

                        # Collect streamed text content
                        collected_text: List[str] = []
                        try:
                            # Prefer the high-level helper if available (Anthropic SDK ≥0.25)
                            if hasattr(stream, "text_stream"):
                                for chunk in stream.text_stream:
                                    collected_text.append(chunk)
                            else:
                                # Generic fallback: inspect each event object
                                for event in stream:
                                    # Text content may live at event.text or event.delta.text
                                    text_piece: Optional[str] = None
                                    if hasattr(event, "text") and event.text:
                                        text_piece = event.text
                                    elif hasattr(event, "delta") and getattr(event.delta, "text", None):
                                        text_piece = event.delta.text  # type: ignore
                                    if text_piece:
                                        collected_text.append(text_piece)
                        finally:
                            # Ensure stream is fully consumed / closed
                            if hasattr(stream, "close"):
                                try:
                                    stream.close()
                                except Exception:
                                    pass
                        return "".join(collected_text)
                    except Exception as stream_err:
                        print(f"❌ Streaming retry failed: {stream_err}")
                        raise stream_err
                # Re-raise any unrelated errors
                raise
        
        # Fallback: Generate without web search
        if web_search_attempted:
            print("📝 Generating response without web search")
        elif self.enable_web_search and not self._web_search_available:
            print("📝 Generating response without web search (connectivity unavailable)")
        
        try:
            if pdf_path and not self.proxy_mode:
                message_params["betas"] = ["files-api-2025-04-14"]
                message = self.client.beta.messages.create(**message_params)
            else:
                # For proxy mode or text-only, use the standard API
                message = self.client.messages.create(**message_params)

            # Handle tool use conversation if needed (fallback might still have tools)
            return self._handle_tool_use_conversation(message, message_params, pdf_path, tool_turn_count=0)
        except Exception as e:
            err_msg = str(e).lower()
            if "streaming is strongly recommended" in err_msg or "with_streaming_response" in err_msg or "long-requests" in err_msg:
                print("⚠️ Large request detected in fallback – retrying with streaming mode enabled ...")
                message_params["stream"] = True
                try:
                    if pdf_path and not self.proxy_mode:
                        message_params["betas"] = ["files-api-2025-04-14"]
                        stream = self.client.beta.messages.create(**message_params)
                    else:
                        stream = self.client.messages.create(**message_params)

                    collected_text: List[str] = []
                    try:
                        if hasattr(stream, "text_stream"):
                            for chunk in stream.text_stream:
                                collected_text.append(chunk)
                        else:
                            for event in stream:
                                text_piece: Optional[str] = None
                                if hasattr(event, "text") and event.text:
                                    text_piece = event.text
                                elif hasattr(event, "delta") and getattr(event.delta, "text", None):
                                    text_piece = event.delta.text  # type: ignore
                                if text_piece:
                                    collected_text.append(text_piece)
                    finally:
                        if hasattr(stream, "close"):
                            try:
                                stream.close()
                            except Exception:
                                pass
                    return "".join(collected_text)
                except Exception as stream_err:
                    print(f"❌ Streaming retry failed in fallback: {stream_err}")
                    raise stream_err
            raise
    
    def _handle_tool_use_conversation(self, message, message_params, pdf_path, tool_turn_count: int = 0):
        """Handle tool use conversation flow properly."""
        # Check if there are any tool use blocks in the response
        tool_use_blocks = []
        text_blocks = []
        
        for content_block in message.content:
            if hasattr(content_block, 'type'):
                if content_block.type == 'tool_use':
                    tool_use_blocks.append(content_block)
                elif content_block.type == 'text':
                    text_blocks.append(content_block.text)
        
        # If no tool use, just return the text
        if not tool_use_blocks:
            return self._extract_text_from_response(message)
        
        # Execute tools and prepare tool results
        tool_results = []
        for tool_block in tool_use_blocks:
            tool_name = tool_block.name
            tool_input = tool_block.input
            tool_use_id = tool_block.id
            
            print(f"🔧 [TOOL] Executing tool: {tool_name} with input: {tool_input}")
            
            if tool_name == 'mcp_generic_math':
                # Generic math router with fallback for simple expressions
                requested_tool = tool_input.get('tool')
                requested_args = tool_input.get('args', {})
                
                # Check if this is a simple expression call to evaluate_expression
                if requested_tool == 'evaluate_expression' and 'expression' in requested_args:
                    expression = requested_args['expression']
                    # Simple expression regex: only digits, +, -, ×, ÷, ^, sqrt, ≤ 15 chars
                    simple_expr_pattern = r'^[0-9+\-×÷^√\s\(\)\.]{1,15}$'
                    if re.match(simple_expr_pattern, expression):
                        print(f"🧮 [MCP] Ignoring simple expression call: {expression} (should be handled by LLM)")
                        # Don't add to tool_results, effectively ignoring this tool call
                        # This prevents incrementing tool_turn_count for simple expressions
                        continue
                
                print(f"🧮 [MCP] Generic math tool routing to: {requested_tool} with args {requested_args}")

                if self._mcp_client and requested_tool:
                    result = self._mcp_client.call_tool(requested_tool, requested_args)
                    if result and result.get('success'):
                        tool_result = result.get('result', '')
                        tool_results.append({
                            "tool_use_id": tool_use_id,
                            "type": "tool_result",
                            "content": str(tool_result)
                        })
                    else:
                        error_msg = result.get('error', 'Unknown error') if result else 'No response'
                        tool_results.append({
                            "tool_use_id": tool_use_id,
                            "type": "tool_result",
                            "content": f"Error: {error_msg}",
                            "is_error": True
                        })
                else:
                    tool_results.append({
                        "tool_use_id": tool_use_id,
                        "type": "tool_result",
                        "content": "MCP client not available or tool not specified",
                        "is_error": True
                    })

            elif tool_name == 'bash':
                # Handle bash tool calls
                command = tool_input.get('command', '')
                print(f"🖥️ [BASH] Claude is using bash tool with command: {command}")
                
                if self._sandbox:
                    result = self.execute_code(command, 'bash')
                    if result:
                        print(f"💻 [BASH] Execution result: {result}")
                        tool_results.append({
                            "tool_use_id": tool_use_id,
                            "type": "tool_result",
                            "content": result
                        })
                    else:
                        print(f"❌ [BASH] Execution failed")
                        tool_results.append({
                            "tool_use_id": tool_use_id,
                            "type": "tool_result",
                            "content": f"Bash execution failed: {command}",
                            "is_error": True
                        })
                else:
                    print(f"❌ [BASH] Tool called but sandbox not available")
                    tool_results.append({
                        "tool_use_id": tool_use_id,
                        "type": "tool_result",
                        "content": f"Bash tool not available: {command}",
                        "is_error": True
                    })
        
        # If we have tool results, continue the conversation
        if tool_results:
            # Stop if the maximum number of tool turns has been reached
            if tool_turn_count >= MAX_TOOL_TURNS - 1:
                print(f"🛑 Max tool turns ({MAX_TOOL_TURNS}) reached. Forcing final answer generation.")
                
                # Build the conversation history
                conversation_messages = message_params["messages"].copy()
                conversation_messages.append({"role": "assistant", "content": message.content})
                conversation_messages.append({"role": "user", "content": tool_results})
                
                # Add a final instruction to generate a response without tools
                conversation_messages.append({
                    "role": "user",
                    "content": "TOOL_LIMIT_REACHED – Please provide your final answer using the tool results above. Do not call any more tools."
                })
                
                # Create new message params for the follow-up, but WITHOUT tools
                final_answer_params = {
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "messages": conversation_messages
                }
                
                if self.system:
                    final_answer_params["system"] = self.system

                # Generate the final response without allowing any more tool calls
                if pdf_path and not self.proxy_mode:
                    final_answer_params["betas"] = ["files-api-2025-04-14"]
                    final_message = self.client.beta.messages.create(**final_answer_params)
                else:
                    final_message = self.client.messages.create(**final_answer_params)
                
                return self._extract_text_from_response(final_message)

            print(f"🔄 [TOOL] Continuing conversation with {len(tool_results)} tool results (turn {tool_turn_count + 1})")
            
            # Build the conversation history
            conversation_messages = message_params["messages"].copy()
            conversation_messages.append({"role": "assistant", "content": message.content})
            conversation_messages.append({"role": "user", "content": tool_results})
            
            # Create new message params for the follow-up
            follow_up_params = message_params.copy()
            follow_up_params["messages"] = conversation_messages
            
            # Generate follow-up response
            if pdf_path and not self.proxy_mode:
                follow_up_params["betas"] = ["files-api-2025-04-14"]
                follow_up_message = self.client.beta.messages.create(**follow_up_params)
            else:
                follow_up_message = self.client.messages.create(**follow_up_params)
            
            # Recursively handle any additional tool calls
            return self._handle_tool_use_conversation(follow_up_message, follow_up_params, pdf_path, tool_turn_count + 1)
        
        # No tool results, return the text
        return self._extract_text_from_response(message)
    
    def _extract_text_from_response(self, message) -> str:
        """Extract text content from Claude's response."""
        text_parts = []
        
        if logger.isEnabledFor(logging.DEBUG):
            print(f"🔍 [DEBUG] Processing {len(message.content)} content blocks")
        
        for i, content_block in enumerate(message.content):
            if logger.isEnabledFor(logging.DEBUG):
                print(f"🔍 [DEBUG] Block {i}: type={getattr(content_block, 'type', 'unknown')}")
            
            if hasattr(content_block, 'type'):
                if content_block.type == 'text':
                    if logger.isEnabledFor(logging.DEBUG):
                        print(f"📝 [DEBUG] Text block: {content_block.text[:100]}...")
                    text_parts.append(content_block.text)
                elif content_block.type == 'tool_use':
                    # Tool use blocks are handled in _handle_tool_use_conversation
                    # This should not happen in final responses
                    print(f"⚠️ [DEBUG] Unexpected tool_use block in final response: {content_block.name}")
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
        
        final_text = ''.join(text_parts)
        if logger.isEnabledFor(logging.DEBUG):
            print(f"🔍 [DEBUG] Final extracted text length: {len(final_text)}")
            print(f"🔍 [DEBUG] Final text preview: {final_text[:200]}...")
        return final_text
    
    def get_name(self) -> str:
        return f"Anthropic ({self.model})"
    
    def clear_file_cache(self):
        """Clear cached file IDs and cleanup all resources."""
        self._file_id_cache.clear()
        print("🗑️ Cleared Anthropic file cache")
        
        # Cleanup sandbox session if exists
        if self._sandbox and hasattr(self._sandbox, 'cleanup_session'):
            self._sandbox.cleanup_session()
            print("🗑️ Cleaned up sandbox session")
        
        # Disconnect this instance's MCP client
        if self._mcp_client:
            self._mcp_client.disconnect()
            self._mcp_client = None
            print("🗑️ Disconnected MCP client")
    
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
    
    @property
    def supports_code_execution(self) -> bool:
        """Indicates that Anthropic provider supports code execution."""
        return self.enable_code_execution
    
    @property
    def supports_mcp(self) -> bool:
        """Indicates that Anthropic provider supports MCP integration."""
        return self.enable_mcp and self._mcp_client is not None
    
    def execute_code(self, code: str, language: str = "python", **kwargs) -> Optional[str]:
        """Execute code in a secure sandboxed environment.
        
        Args:
            code: The code to execute
            language: Programming language (default: python)
            **kwargs: Additional execution parameters
            
        Returns:
            Execution result or None if not supported
        """
        if not self.supports_code_execution or not self._sandbox:
            return None
        
        try:
            # Execute code in the secure sandbox
            result = self._sandbox.execute_code(code, language, **kwargs)
            
            if result['success']:
                output_parts = []
                if result.get('stdout'):
                    output_parts.append(f"STDOUT:\n{result['stdout']}")
                if result.get('stderr'):
                    output_parts.append(f"STDERR:\n{result['stderr']}")
                
                if output_parts:
                    return '\n'.join(output_parts)
                else:
                    return "Code executed successfully with no output"
            else:
                error_msg = result.get('error', 'Unknown error')
                if result.get('stderr'):
                    error_msg += f"\nSTDERR:\n{result['stderr']}"
                return f"Execution failed: {error_msg}"
                
        except Exception as e:
            logger.error(f"Secure code execution failed: {e}")
            return f"Sandbox execution error: {str(e)}"
    
    def call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[str]:
        """Call an MCP tool by name.
        
        Args:
            tool_name: Name of the MCP tool to call
            arguments: Arguments to pass to the tool
            
        Returns:
            Tool result or None if failed
        """
        if not self._mcp_client:
            return None
        
        try:
            result = self._mcp_client.call_tool(tool_name, arguments)
            if result and result.get('success'):
                return str(result.get('result', ''))
            else:
                error_msg = result.get('error', 'Unknown error') if result else 'No response'
                logger.error(f"MCP tool '{tool_name}' failed: {error_msg}")
                return f"Tool error: {error_msg}"
        except Exception as e:
            logger.error(f"MCP tool call failed: {e}")
            return f"Tool call error: {str(e)}"
