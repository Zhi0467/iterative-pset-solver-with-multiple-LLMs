#!/usr/bin/env python3
"""
Test to see if Claude will actually use the bash tool when prompted explicitly
"""

import os
import sys
from dotenv import load_dotenv

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_providers.anthropic_provider import AnthropicProvider

load_dotenv()

def test_explicit_bash_usage():
    """Test if Claude will use bash tool when explicitly instructed"""
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not found")
        return False
    
    # Create provider with code execution enabled
    provider = AnthropicProvider(
        api_key=api_key,
        enable_code_execution=True,
        enable_mcp=False,
        enable_web_search=False
    )
    
    # Very explicit prompt about using bash tool
    test_prompt = """
    You have access to a bash tool. Please use it to:
    1. Create a simple Python file with a DFS function
    2. Test the function
    
    IMPORTANT: You MUST use the bash tool to write and test the code.
    
    Example:
    - Use bash tool to create file: echo "def dfs(): pass" > test.py
    - Use bash tool to test: python test.py
    
    Please implement depth-first search in Python using the bash tool.
    """
    
    print("🔍 Testing explicit bash tool usage...")
    print("Prompt:", test_prompt)
    print("\n" + "="*50)
    
    try:
        response = provider.generate(test_prompt)
        print("Response:", response)
        
        # Check if bash tool was used (should see the debug messages)
        if "🖥️ Claude is using bash tool" in response:
            print("✅ Claude used bash tool!")
            return True
        else:
            print("❌ Claude did not use bash tool")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_explicit_bash_usage()
    sys.exit(0 if success else 1)