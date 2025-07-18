#!/usr/bin/env python3
"""Test tool use conversation flow"""

import os
import sys
from utils.config import LLMConfig
from llm_providers.anthropic_provider import AnthropicProvider

def test_tool_conversation():
    """Test the tool use conversation flow with a simple math problem"""
    config = LLMConfig()
    
    # Get Anthropic config
    anthropic_config = config.get_llm_config('anthropic')
    
    # Create provider
    provider = AnthropicProvider(
        api_key=anthropic_config['api_key'],
        model=anthropic_config['model'],
        max_tokens=anthropic_config['max_tokens'],
        temperature=anthropic_config['temperature'],
        enable_web_search=anthropic_config['enable_web_search'],
        enable_code_execution=anthropic_config['enable_code_execution'],
        enable_mcp=anthropic_config['enable_mcp']
    )
    
    # Test simple math problem
    prompt = "What is 2 + 6? Use the MCP math tools to calculate this and show your work."
    
    print("🧪 Testing tool conversation flow...")
    print(f"📝 Prompt: {prompt}")
    
    try:
        result = provider.generate(prompt)
        print(f"📊 Result: {result}")
        return result
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    test_tool_conversation()