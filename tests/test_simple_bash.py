#!/usr/bin/env python3
"""
Simple test to isolate the bash tool issue
"""

import os
import sys
from dotenv import load_dotenv

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_providers.anthropic_provider import AnthropicProvider

load_dotenv()

def test_simple_bash():
    """Test simple bash command"""
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not found")
        return False
    
    # Create provider with minimal configuration
    provider = AnthropicProvider(
        api_key=api_key,
        enable_code_execution=True,
        enable_mcp=False,
        enable_web_search=False,
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        temperature=0.1
    )
    
    # Test direct sandbox execution first
    print("🔍 Testing direct sandbox execution...")
    if provider._sandbox:
        result = provider._sandbox.execute_code("echo 'Hello World'", "bash")
        print(f"Direct sandbox result: {result}")
    
    # Test provider execute_code method
    print("\n🔍 Testing provider execute_code method...")
    result = provider.execute_code("echo 'Hello World'", "bash")
    print(f"Provider execute_code result: {result}")
    
    # Test very simple prompt
    print("\n🔍 Testing simple prompt...")
    simple_prompt = "Please use the bash tool to run: echo 'test'"
    
    try:
        response = provider.generate(simple_prompt)
        print(f"Response: {response}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_simple_bash()