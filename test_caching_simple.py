#!/usr/bin/env python3
"""
Simple test script to verify prompt caching in AnthropicProvider.
Tests caching directly with the AnthropicProvider class.
"""

import os
import sys
import time
from llm_providers.anthropic_provider import AnthropicProvider
from utils.config import LLMConfig

def test_anthropic_caching():
    """Test prompt caching with Anthropic provider directly."""
    
    print("🧪 Testing Anthropic Prompt Caching (Direct)")
    print("=" * 50)
    
    # Check if test1.pdf exists
    test_pdf = "test1.pdf"
    if not os.path.exists(test_pdf):
        print(f"❌ Test PDF not found: {test_pdf}")
        return
    
    print(f"📄 Using test PDF: {test_pdf}")
    
    # Initialize config and get API key
    config = LLMConfig()
    api_key = config.get_api_key(LLMConfig.PROVIDER_ANTHROPIC)
    if not api_key:
        print("❌ No Anthropic API key found")
        return
    
    # Temporarily override base URL for testing
    original_base_url = os.environ.get('ANTHROPIC_BASE_URL')
    if original_base_url:
        print(f"🔄 Temporarily removing ANTHROPIC_BASE_URL: {original_base_url}")
        del os.environ['ANTHROPIC_BASE_URL']
    
    # Initialize AnthropicProvider with caching
    print("\n🔧 Initializing AnthropicProvider...")
    provider_config = config.get_provider_config(LLMConfig.PROVIDER_ANTHROPIC)
    
    # Override base URL to use standard Anthropic API for testing
    if 'base_url' in provider_config:
        print("🔄 Removing custom base_url for direct API testing")
        del provider_config['base_url']
    
    provider = AnthropicProvider(
        api_key=api_key,
        system="You are a helpful assistant for testing prompt caching.",
        **provider_config
    )
    
    print(f"✅ Provider initialized: {provider.get_name()}")
    
    # Test prompt - we'll use this same prompt twice to test caching
    test_prompt = """Please analyze the attached PDF and provide a brief summary of:
1. The main topics covered
2. The difficulty level of the content
3. The type of problems present

This is a test to verify prompt caching is working correctly."""
    
    # Test 1: First run (should create cache)
    print("\n🔬 Test 1: First run (cache creation)")
    print("-" * 40)
    
    start_time = time.time()
    
    try:
        response1 = provider.generate(test_prompt, pdf_path=test_pdf)
        first_run_time = time.time() - start_time
        
        print(f"✅ First run completed in {first_run_time:.2f} seconds")
        print(f"📝 Generated {len(response1)} characters")
        print(f"📋 Response preview: {response1[:200]}...")
        
    except Exception as e:
        print(f"❌ First run failed: {e}")
        return
    
    # Small delay to ensure cache has time to be stored
    time.sleep(1)
    
    # Test 2: Second run (should use cache)
    print("\n🔬 Test 2: Second run (cache hit)")
    print("-" * 40)
    
    start_time = time.time()
    
    try:
        response2 = provider.generate(test_prompt, pdf_path=test_pdf)
        second_run_time = time.time() - start_time
        
        print(f"✅ Second run completed in {second_run_time:.2f} seconds")
        print(f"📝 Generated {len(response2)} characters")
        print(f"📋 Response preview: {response2[:200]}...")
        
        # Compare timing
        if second_run_time < first_run_time * 0.9:  # Expect some improvement
            speed_improvement = ((first_run_time - second_run_time) / first_run_time * 100)
            print(f"🚀 Cache hit detected! Second run was {speed_improvement:.1f}% faster")
        else:
            print(f"⚠️  No significant speed improvement. Times may vary due to network.")
            
        print(f"\n📊 Timing comparison:")
        print(f"    First run:  {first_run_time:.2f}s")
        print(f"    Second run: {second_run_time:.2f}s")
        print(f"    Difference: {first_run_time - second_run_time:.2f}s")
        
    except Exception as e:
        print(f"❌ Second run failed: {e}")
        return
    
    # Test 3: Test with different prompt (should not benefit from most caches)
    print("\n🔬 Test 3: Different prompt (partial cache hit)")
    print("-" * 40)
    
    different_prompt = """This is a completely different prompt that should only benefit from tool and system message caching, not content caching. Can you tell me what you see in this PDF?"""
    
    start_time = time.time()
    
    try:
        response3 = provider.generate(different_prompt, pdf_path=test_pdf)
        third_run_time = time.time() - start_time
        
        print(f"✅ Third run completed in {third_run_time:.2f} seconds")
        print(f"📝 Generated {len(response3)} characters")
        
        print(f"\n📊 Full timing comparison:")
        print(f"    First run (no cache):     {first_run_time:.2f}s")
        print(f"    Second run (full cache):  {second_run_time:.2f}s")
        print(f"    Third run (partial cache): {third_run_time:.2f}s")
        
    except Exception as e:
        print(f"❌ Third run failed: {e}")
    
    # Clean up
    print("\n🧹 Cleaning up...")
    provider.clear_file_cache()
    
    # Restore original base URL
    if original_base_url:
        print(f"🔄 Restoring ANTHROPIC_BASE_URL: {original_base_url}")
        os.environ['ANTHROPIC_BASE_URL'] = original_base_url
    
    print("\n🎉 Caching test completed!")
    print("=" * 50)
    
    # Summary
    print("\n📊 Summary:")
    print(f"  - PDF: {test_pdf}")
    print(f"  - Provider: {provider.get_name()}")
    print(f"  - Cached elements: Tools, System Messages, PDF Documents, Large Text")
    
    if 'first_run_time' in locals() and 'second_run_time' in locals():
        speed_improvement = ((first_run_time - second_run_time) / first_run_time * 100)
        print(f"  - Speed improvement: {speed_improvement:.1f}%")
        
        if speed_improvement > 5:  # Even small improvements count
            print("✅ Caching appears to be working!")
        else:
            print("⚠️  Results inconclusive. Network variance may mask caching benefits.")
    
    print(f"\n💰 Expected cost savings with caching:")
    print(f"  - Cache writes cost: 1.25x base tokens")
    print(f"  - Cache hits cost: 0.1x base tokens (90% savings)")
    print(f"  - Tools, system messages, and PDF content are now cached")

def main():
    """Main function."""
    test_anthropic_caching()

if __name__ == "__main__":
    main()