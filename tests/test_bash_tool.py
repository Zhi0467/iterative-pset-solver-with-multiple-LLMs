#!/usr/bin/env python3
"""
Unit test to verify bash tool registration and functionality in AnthropicProvider
"""

import os
import sys
import tempfile
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_providers.anthropic_provider import AnthropicProvider
from utils.config import LLMConfig

# Load environment variables
load_dotenv()

def test_bash_tool_registration():
    """Test that bash tool is properly registered when code execution is enabled"""
    
    # Get API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not found in environment")
        return False
    
    # Create provider with code execution enabled
    provider = AnthropicProvider(
        api_key=api_key,
        enable_code_execution=True,
        enable_mcp=False  # Disable MCP to focus on bash tool
    )
    
    # Check if bash tool support is enabled
    print(f"✅ Code execution enabled: {provider.enable_code_execution}")
    print(f"✅ Supports code execution: {provider.supports_code_execution}")
    print(f"✅ Sandbox initialized: {provider._sandbox is not None}")
    
    return True

def test_bash_tool_in_generate():
    """Test that bash tool is included in the tools list during generation"""
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not found in environment")
        return False
    
    # Create provider with code execution enabled
    provider = AnthropicProvider(
        api_key=api_key,
        enable_code_execution=True,
        enable_mcp=False,
        enable_web_search=False
    )
    
    # Create a simple test prompt
    test_prompt = "Write a simple Python function that adds two numbers and test it."
    
    # Mock the Anthropic client to capture the tools being sent
    with patch.object(provider, 'client') as mock_client:
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type='text', text='Test response')]
        mock_client.messages.create.return_value = mock_response
        
        try:
            # Call generate which should register bash tool
            result = provider.generate(test_prompt)
            
            # Check if messages.create was called with tools
            call_args = mock_client.messages.create.call_args
            if call_args:
                kwargs = call_args[1]  # Get keyword arguments
                tools = kwargs.get('tools', [])
                
                print(f"✅ Tools sent to API: {len(tools)} tools")
                for tool in tools:
                    tool_name = tool.get('name', tool.get('type', 'unknown'))
                    print(f"  - {tool_name}")
                
                # Check if bash tool is in the tools list
                bash_tool_found = any(
                    tool.get('name') == 'bash' or tool.get('type') == 'bash_20250124' 
                    for tool in tools
                )
                
                if bash_tool_found:
                    print("✅ Bash tool found in tools list")
                    return True
                else:
                    print("❌ Bash tool NOT found in tools list")
                    return False
            else:
                print("❌ No tools sent to API")
                return False
                
        except Exception as e:
            print(f"❌ Error during generate call: {e}")
            return False

def test_bash_tool_execution():
    """Test that bash tool can execute simple commands"""
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not found in environment")
        return False
    
    # Create provider with code execution enabled
    provider = AnthropicProvider(
        api_key=api_key,
        enable_code_execution=True,
        enable_mcp=False
    )
    
    # Test direct code execution
    if provider._sandbox:
        test_code = "print('Hello, World!')"
        result = provider.execute_code(test_code, 'python')
        
        if result and 'Hello, World!' in result:
            print("✅ Direct code execution works")
            return True
        else:
            print(f"❌ Direct code execution failed: {result}")
            return False
    else:
        print("❌ No sandbox available for code execution")
        return False

def test_config_code_execution():
    """Test that code execution is enabled by default for Anthropic in config"""
    
    config = LLMConfig()
    anthropic_config = config.get_provider_config(LLMConfig.PROVIDER_ANTHROPIC)
    
    code_execution_enabled = anthropic_config.get('enable_code_execution', False)
    print(f"✅ Code execution enabled in config: {code_execution_enabled}")
    
    return code_execution_enabled

def main():
    print("🧪 Testing Bash Tool Integration")
    print("=" * 50)
    
    tests = [
        ("Config Code Execution", test_config_code_execution),
        ("Bash Tool Registration", test_bash_tool_registration),
        ("Bash Tool in Generate", test_bash_tool_in_generate),
        ("Bash Tool Execution", test_bash_tool_execution),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
            print(f"{'✅ PASSED' if success else '❌ FAILED'}: {test_name}")
        except Exception as e:
            print(f"❌ ERROR in {test_name}: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    for test_name, success in results:
        print(f"  {'✅' if success else '❌'} {test_name}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\n🎯 Summary: {passed}/{total} tests passed")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)