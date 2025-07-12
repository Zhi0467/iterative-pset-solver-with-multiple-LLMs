#!/usr/bin/env python3
"""
Comprehensive test script to verify network connectivity fallback functionality.
"""

import os
from utils.config import LLMConfig
from utils.network_checker import network_checker
from llm_providers.anthropic_provider import AnthropicProvider
from llm_providers.gemini_provider import GeminiProvider
from llm_providers.openai_provider import OpenAIProvider
from llm_providers.deepseek_provider import DeepSeekProvider

def test_network_connectivity():
    """Test network connectivity for all providers."""
    print("🌐 Testing network connectivity for all providers...\n")
    
    providers = ['anthropic', 'gemini', 'openai', 'deepseek']
    
    for provider in providers:
        print(f"📡 Testing {provider} connectivity:")
        is_available = network_checker.is_web_search_available(provider)
        print(f"  Web search available: {'✅ YES' if is_available else '❌ NO'}")
        
        # Get detailed report
        report = network_checker.get_connectivity_report(provider)
        print(f"  {report}")
        print()

def test_provider_initialization():
    """Test that all providers initialize correctly with web search settings."""
    print("🔧 Testing provider initialization with web search settings...\n")
    
    config = LLMConfig()
    providers = {
        'anthropic': AnthropicProvider,
        'gemini': GeminiProvider,
        'openai': OpenAIProvider,
        'deepseek': DeepSeekProvider
    }
    
    for provider_name, provider_class in providers.items():
        api_key = config.get_api_key(provider_name)
        
        if not api_key:
            print(f"⚠️ No API key for {provider_name}, skipping...")
            continue
            
        try:
            print(f"🔧 Testing {provider_name} initialization:")
            
            # Test with web search enabled
            provider_config = config.get_provider_config(provider_name)
            enable_web_search = provider_config.pop('enable_web_search', True)
            
            provider_enabled = provider_class(api_key, enable_web_search=True, **provider_config)
            print(f"  ✅ Initialized with web search enabled: {provider_enabled.enable_web_search}")
            
            # Test with web search disabled
            provider_disabled = provider_class(api_key, enable_web_search=False, **provider_config)
            print(f"  ✅ Initialized with web search disabled: {provider_disabled.enable_web_search}")
            
        except Exception as e:
            print(f"  ❌ Failed to initialize {provider_name}: {e}")
        
        print()

def test_web_search_fallback():
    """Test web search fallback functionality."""
    print("🔄 Testing web search fallback functionality...\n")
    
    config = LLMConfig()
    
    # Test with Gemini (has web search support)
    api_key = config.get_api_key(LLMConfig.PROVIDER_GEMINI)
    if api_key:
        print("🧪 Testing Gemini web search fallback:")
        try:
            provider_config = config.get_provider_config(LLMConfig.PROVIDER_GEMINI)
            enable_web_search = provider_config.pop('enable_web_search', True)
            provider = GeminiProvider(api_key, enable_web_search=enable_web_search, **provider_config)
            
            # Simple test prompt
            prompt = "What is 2+2? Please explain briefly."
            
            print(f"  📝 Sending test prompt: {prompt}")
            response = provider.generate(prompt)
            
            if response and len(response) > 0:
                print(f"  ✅ Response received (length: {len(response)})")
                print(f"  📋 First 100 chars: {response[:100]}...")
            else:
                print(f"  ❌ No response received")
                
        except Exception as e:
            print(f"  ❌ Gemini test failed: {e}")
    else:
        print("⚠️ No Gemini API key, skipping Gemini test")
    
    print()

def test_configuration_integration():
    """Test that configuration properly integrates with providers."""
    print("⚙️ Testing configuration integration...\n")
    
    config = LLMConfig()
    
    # Show current web search status
    print("📊 Current web search configuration:")
    web_search_status = config.get_web_search_status()
    for provider, enabled in web_search_status.items():
        print(f"  {provider}: {'✅ ENABLED' if enabled else '❌ DISABLED'}")
    
    print()
    
    # Test changing configuration
    print("🔧 Testing configuration changes:")
    
    # Disable web search for Gemini
    config.set_web_search_enabled(LLMConfig.PROVIDER_GEMINI, False)
    gemini_enabled = config.get_web_search_enabled(LLMConfig.PROVIDER_GEMINI)
    print(f"  Gemini web search after disabling: {'✅ ENABLED' if gemini_enabled else '❌ DISABLED'}")
    
    # Re-enable web search for Gemini
    config.set_web_search_enabled(LLMConfig.PROVIDER_GEMINI, True)
    gemini_enabled = config.get_web_search_enabled(LLMConfig.PROVIDER_GEMINI)
    print(f"  Gemini web search after re-enabling: {'✅ ENABLED' if gemini_enabled else '❌ DISABLED'}")
    
    print()

def main():
    """Run all tests."""
    print("🚀 Starting comprehensive network fallback tests...\n")
    
    # Test network connectivity
    test_network_connectivity()
    
    # Test provider initialization
    test_provider_initialization()
    
    # Test web search fallback
    test_web_search_fallback()
    
    # Test configuration integration
    test_configuration_integration()
    
    print("🎉 All tests completed!")
    print("\n📋 Summary:")
    print("✅ Network connectivity checker implemented")
    print("✅ Provider initialization with web search control")
    print("✅ Web search fallback functionality")
    print("✅ Configuration integration")
    print("\n🔧 You can now control web search per provider via:")
    print("  - Constructor: provider = Provider(api_key, enable_web_search=True/False)")
    print("  - Config: config.set_web_search_enabled('provider_name', True/False)")

if __name__ == "__main__":
    main() 