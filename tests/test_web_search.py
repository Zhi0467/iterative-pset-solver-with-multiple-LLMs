#!/usr/bin/env python3
"""
Test script to verify web search functionality with Claude and Gemini providers.
"""

import os
from utils.config import LLMConfig
from llm_providers.anthropic_provider import AnthropicProvider
from llm_providers.gemini_provider import GeminiProvider

def test_claude_web_search():
    """Test Claude's web search capability."""
    print("🧪 Testing Claude web search...")
    
    config = LLMConfig()
    api_key = config.get_api_key(LLMConfig.PROVIDER_ANTHROPIC)
    
    if not api_key:
        print("❌ No Anthropic API key found. Skipping Claude test.")
        return False
    
    try:
        # Initialize provider with web search enabled
        provider_config = config.get_provider_config(LLMConfig.PROVIDER_ANTHROPIC)
        provider_config['system'] = "You are a helpful assistant."
        
        # Add base URL if available
        base_url = config.get_base_url(LLMConfig.PROVIDER_ANTHROPIC)
        if base_url:
            provider_config['base_url'] = base_url
            provider_config['proxy_mode'] = True
        
        # Extract enable_web_search from config
        enable_web_search = provider_config.pop('enable_web_search', True)
        provider = AnthropicProvider(api_key, enable_web_search=enable_web_search, **provider_config)
        
        # Test prompt that should trigger web search
        prompt = "What is the current price of Bitcoin? Please search for the latest information."
        
        print(f"🔍 Sending prompt: {prompt}")
        response = provider.generate(prompt)
        
        print(f"✅ Claude response (first 200 chars): {response[:200]}...")
        return True
        
    except Exception as e:
        print(f"❌ Claude test failed: {e}")
        return False

def test_gemini_web_search():
    """Test Gemini's Google Search grounding capability."""
    print("🧪 Testing Gemini web search...")
    
    config = LLMConfig()
    api_key = config.get_api_key(LLMConfig.PROVIDER_GEMINI)
    
    if not api_key:
        print("❌ No Gemini API key found. Skipping Gemini test.")
        return False
    
    try:
        # Initialize provider with web search enabled
        provider_config = config.get_provider_config(LLMConfig.PROVIDER_GEMINI)
        
        # Extract enable_web_search from config
        enable_web_search = provider_config.pop('enable_web_search', True)
        provider = GeminiProvider(api_key, enable_web_search=enable_web_search, **provider_config)
        
        # Test prompt that should trigger web search
        prompt = "Who won the 2024 UEFA European Championship? Please search for the latest results."
        
        print(f"🔍 Sending prompt: {prompt}")
        response = provider.generate(prompt)
        
        print(f"✅ Gemini response (first 200 chars): {response[:200]}...")
        return True
        
    except Exception as e:
        print(f"❌ Gemini test failed: {e}")
        return False

def test_web_search_disabled():
    """Test that web search can be disabled."""
    print("🧪 Testing web search disabled...")
    
    config = LLMConfig()
    api_key = config.get_api_key(LLMConfig.PROVIDER_GEMINI)
    
    if not api_key:
        print("❌ No Gemini API key found. Skipping disabled test.")
        return False
    
    try:
        # Initialize provider with web search disabled
        provider_config = config.get_provider_config(LLMConfig.PROVIDER_GEMINI)
        # Remove enable_web_search from config to avoid conflict
        provider_config.pop('enable_web_search', None)
        provider = GeminiProvider(api_key, enable_web_search=False, **provider_config)
        
        # Test prompt that would normally trigger web search
        prompt = "Who won the 2024 UEFA European Championship? Please search for the latest results."
        
        print(f"🔍 Sending prompt with web search disabled: {prompt}")
        response = provider.generate(prompt)
        
        print(f"✅ Gemini response without web search (first 200 chars): {response[:200]}...")
        return True
        
    except Exception as e:
        print(f"❌ Disabled web search test failed: {e}")
        return False

def main():
    """Run web search tests for both providers."""
    print("🚀 Starting web search functionality tests...\n")
    
    claude_success = test_claude_web_search()
    print()
    gemini_success = test_gemini_web_search()
    print()
    disabled_success = test_web_search_disabled()
    
    print("\n📋 Test Results:")
    print(f"Claude web search: {'✅ PASS' if claude_success else '❌ FAIL'}")
    print(f"Gemini web search: {'✅ PASS' if gemini_success else '❌ FAIL'}")
    print(f"Web search disabled: {'✅ PASS' if disabled_success else '❌ FAIL'}")
    
    if claude_success or gemini_success:
        print("\n🎉 At least one provider supports web search!")
        if disabled_success:
            print("✅ Web search can be properly disabled when needed.")
    else:
        print("\n⚠️ No providers successfully demonstrated web search.")

if __name__ == "__main__":
    main() 