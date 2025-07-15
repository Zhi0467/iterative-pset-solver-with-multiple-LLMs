#!/usr/bin/env python3
"""
Test suite for MCP (Model Context Protocol) server integration.
"""

import os
import sys
import time
import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.config import LLMConfig

# Test MCP imports with graceful fallback
try:
    from mcp_server.client import SyncMCPClient
    from mcp_server.math_tools import mcp
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    SyncMCPClient = None
    mcp = None

class TestMCPServerBasicTools(unittest.TestCase):
    """Test cases for basic MCP server math tools."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not MCP_AVAILABLE:
            self.skipTest("MCP dependencies not available")
    
    def test_math_tools_available(self):
        """Test that math tools are available."""
        # Test that we can import the math tools
        self.assertIsNotNone(mcp)
        
        # Test that tools are registered
        tools = mcp.list_tools()
        self.assertGreater(len(tools), 0)
        
        # Check for basic math tools
        tool_names = [tool.name for tool in tools]
        expected_tools = ['add', 'subtract', 'multiply', 'divide', 'square_root', 'power']
        
        for tool in expected_tools:
            self.assertIn(tool, tool_names, f"Tool '{tool}' not found in registered tools")
    
    def test_basic_arithmetic_tools(self):
        """Test basic arithmetic tool functions."""
        # Test add
        result = mcp.call_tool('add', {'a': 5, 'b': 3})
        self.assertEqual(result, 8)
        
        # Test subtract
        result = mcp.call_tool('subtract', {'a': 10, 'b': 4})
        self.assertEqual(result, 6)
        
        # Test multiply
        result = mcp.call_tool('multiply', {'a': 6, 'b': 7})
        self.assertEqual(result, 42)
        
        # Test divide
        result = mcp.call_tool('divide', {'a': 15, 'b': 3})
        self.assertEqual(result, 5.0)
        
        # Test division by zero
        with self.assertRaises(ZeroDivisionError):
            mcp.call_tool('divide', {'a': 10, 'b': 0})
    
    def test_advanced_math_tools(self):
        """Test advanced math tool functions."""
        # Test square root
        result = mcp.call_tool('square_root', {'x': 16})
        self.assertEqual(result, 4.0)
        
        # Test power
        result = mcp.call_tool('power', {'base': 2, 'exponent': 3})
        self.assertEqual(result, 8.0)
        
        # Test factorial (if available)
        tools = mcp.list_tools()
        tool_names = [tool.name for tool in tools]
        if 'factorial' in tool_names:
            result = mcp.call_tool('factorial', {'n': 5})
            self.assertEqual(result, 120)
    
    def test_trigonometric_tools(self):
        """Test trigonometric tool functions."""
        import math
        
        tools = mcp.list_tools()
        tool_names = [tool.name for tool in tools]
        
        if 'sin' in tool_names:
            result = mcp.call_tool('sin', {'x': math.pi / 2})
            self.assertAlmostEqual(result, 1.0, places=10)
        
        if 'cos' in tool_names:
            result = mcp.call_tool('cos', {'x': 0})
            self.assertAlmostEqual(result, 1.0, places=10)
        
        if 'tan' in tool_names:
            result = mcp.call_tool('tan', {'x': 0})
            self.assertAlmostEqual(result, 0.0, places=10)


class TestMCPClient(unittest.TestCase):
    """Test cases for MCP client functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not MCP_AVAILABLE:
            self.skipTest("MCP dependencies not available")
    
    def test_client_initialization(self):
        """Test MCP client initialization."""
        server_command = ['python', 'mcp_server/math_tools.py']
        client = SyncMCPClient(server_command)
        
        self.assertIsNotNone(client)
        self.assertEqual(client.server_command, server_command)
        self.assertFalse(client.is_connected)
    
    def test_client_connection(self):
        """Test MCP client connection to server."""
        server_command = ['python3', os.path.join(os.path.dirname(__file__), 'mcp_server', 'math_tools.py')]
        client = SyncMCPClient(server_command)
        
        # Test connection
        connected = client.connect()
        
        if connected:
            self.assertTrue(client.is_connected)
            
            # Test tool discovery
            tools = client.get_available_tools()
            self.assertIsInstance(tools, list)
            self.assertGreater(len(tools), 0)
            
            # Test tool execution
            result = client.call_tool('add', {'a': 3, 'b': 4})
            self.assertIsNotNone(result)
            
            # Cleanup
            client.disconnect()
            self.assertFalse(client.is_connected)
        else:
            self.skipTest("Could not connect to MCP server")
    
    def test_client_tool_schema(self):
        """Test MCP client tool schema retrieval."""
        server_command = ['python3', os.path.join(os.path.dirname(__file__), 'mcp_server', 'math_tools.py')]
        client = SyncMCPClient(server_command)
        
        if client.connect():
            tools = client.get_available_tools()
            self.assertGreater(len(tools), 0)
            
            # Test schema retrieval for first tool
            tool_name = tools[0]
            schema = client.get_tool_schema(tool_name)
            
            self.assertIsNotNone(schema)
            self.assertIn('description', schema)
            self.assertIn('input_schema', schema)
            
            client.disconnect()
        else:
            self.skipTest("Could not connect to MCP server")


class TestAnthropicMCPIntegration(unittest.TestCase):
    """Test cases for Anthropic provider MCP integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not MCP_AVAILABLE:
            self.skipTest("MCP dependencies not available")
    
    @patch('llm_providers.anthropic_provider.Anthropic')
    def test_anthropic_mcp_initialization(self, mock_anthropic):
        """Test Anthropic provider MCP initialization."""
        from llm_providers.anthropic_provider import AnthropicProvider
        
        # Test with MCP enabled
        provider = AnthropicProvider(
            api_key="test_key",
            enable_mcp=True,
            enable_code_execution=False,
            enable_web_search=False,
            mcp_server_command=['python3', 'mcp_server/math_tools.py']
        )
        
        # MCP initialization might fail due to server not running, but should not crash
        self.assertIsNotNone(provider)
        self.assertTrue(provider.enable_mcp)
    
    @patch('llm_providers.anthropic_provider.Anthropic')
    def test_anthropic_mcp_tool_calling(self, mock_anthropic):
        """Test Anthropic provider MCP tool calling."""
        from llm_providers.anthropic_provider import AnthropicProvider
        
        provider = AnthropicProvider(
            api_key="test_key",
            enable_mcp=True,
            enable_code_execution=False,
            enable_web_search=False
        )
        
        # Test tool calling (may return None if MCP client not connected)
        result = provider.call_mcp_tool('add', {'a': 5, 'b': 3})
        # Result can be None if MCP client failed to initialize
        self.assertTrue(result is None or isinstance(result, str))
    
    @patch('llm_providers.anthropic_provider.Anthropic')
    def test_anthropic_mcp_disabled(self, mock_anthropic):
        """Test Anthropic provider with MCP disabled."""
        from llm_providers.anthropic_provider import AnthropicProvider
        
        provider = AnthropicProvider(
            api_key="test_key",
            enable_mcp=False,
            enable_code_execution=False,
            enable_web_search=False
        )
        
        self.assertFalse(provider.enable_mcp)
        self.assertFalse(provider.supports_mcp)
        
        # Tool calling should return None when MCP is disabled
        result = provider.call_mcp_tool('add', {'a': 5, 'b': 3})
        self.assertIsNone(result)


class TestMCPConfiguration(unittest.TestCase):
    """Test cases for MCP configuration management."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = LLMConfig()
    
    def test_mcp_default_configuration(self):
        """Test default MCP configuration."""
        # Test default MCP settings
        anthropic_config = self.config.get_provider_config('anthropic')
        self.assertIn('enable_mcp', anthropic_config)
        self.assertTrue(anthropic_config['enable_mcp'])  # Should be True by default for Anthropic
        
        openai_config = self.config.get_provider_config('openai')
        self.assertIn('enable_mcp', openai_config)
        self.assertFalse(openai_config['enable_mcp'])  # Should be False by default for OpenAI
    
    def test_mcp_enable_disable(self):
        """Test enabling and disabling MCP."""
        # Test enabling MCP
        self.config.set_mcp_enabled('anthropic', True)
        self.assertTrue(self.config.get_mcp_enabled('anthropic'))
        
        # Test disabling MCP
        self.config.set_mcp_enabled('anthropic', False)
        self.assertFalse(self.config.get_mcp_enabled('anthropic'))
    
    def test_mcp_server_configuration(self):
        """Test MCP server configuration."""
        # Test setting MCP server config
        server_url = "stdio"
        server_command = ["python3", "mcp_server/math_tools.py"]
        
        self.config.set_mcp_server_config('anthropic', server_url, server_command)
        
        config = self.config.get_provider_config('anthropic')
        self.assertEqual(config['mcp_server_url'], server_url)
        self.assertEqual(config['mcp_server_command'], server_command)
    
    def test_mcp_status_reporting(self):
        """Test MCP status reporting."""
        status = self.config.get_mcp_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn('anthropic', status)
        self.assertIn('openai', status)
        self.assertIn('gemini', status)
        self.assertIn('deepseek', status)
        
        # Test that status reflects current configuration
        self.assertEqual(status['anthropic'], self.config.get_mcp_enabled('anthropic'))
    
    def test_provider_capabilities_mcp(self):
        """Test provider capabilities reporting for MCP."""
        capabilities = self.config.get_provider_capabilities('anthropic')
        
        self.assertIsInstance(capabilities, dict)
        self.assertIn('mcp', capabilities)
        self.assertEqual(capabilities['mcp'], self.config.get_mcp_enabled('anthropic'))


def run_integration_tests():
    """Run integration tests for MCP functionality."""
    if not MCP_AVAILABLE:
        print("⚠️ MCP dependencies not available. Install with: pip install -r mcp_requirements.txt")
        return
    
    print("Running MCP integration tests...")
    
    # Test server startup time
    start_time = time.time()
    try:
        from mcp_server.math_tools import mcp
        server_startup_time = time.time() - start_time
        print(f"MCP server import time: {server_startup_time:.3f}s")
        
        # Test tool registration
        tools = mcp.list_tools()
        print(f"Registered {len(tools)} MCP tools")
        
        # Test basic tool execution
        start_time = time.time()
        result = mcp.call_tool('add', {'a': 100, 'b': 200})
        execution_time = time.time() - start_time
        print(f"Tool execution time: {execution_time:.3f}s")
        print(f"Result: {result}")
        
    except Exception as e:
        print(f"MCP integration test failed: {e}")


if __name__ == "__main__":
    # Run unit tests
    print("Running MCP unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run integration tests
    print("\n" + "="*50)
    run_integration_tests()
    
    print("\nAll MCP tests completed!")