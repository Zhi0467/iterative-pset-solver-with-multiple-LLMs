#!/usr/bin/env python3
"""
Main entry point for the auto-pset solver.
Processes problem sets using an orchestrated workflow of LLM providers.
"""

import os
import sys
import argparse
from processors.orchestrator import Orchestrator
from utils.config import LLMConfig
# Add project root to the Python path to allow absolute imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)



def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process problem sets using an orchestrated workflow of LLM providers."
    )
    
    parser.add_argument(
        "pdf_paths",
        nargs="+",
        help="One or more paths to PDF files to process"
    )
    
    # PDF mode is fixed to direct upload; no CLI option needed
    
    return parser.parse_args()

def validate_environment() -> None:
    """Validate that required API keys are present."""
    config = LLMConfig()
    
    # Check for OpenAI (used for both strategist and analyst)
    if not config.get_api_key(LLMConfig.PROVIDER_OPENAI):
        print("⚠️ Warning: OpenAI API key not found. The orchestrator requires GPT-4.")
        print("Please set OPENAI_API_KEY environment variable.")
        sys.exit(1)
    
    # Check for other providers that might be used
    available_providers = config.get_available_providers()
    print("\n🔑 Available LLM Providers:")
    for provider, has_key in available_providers.items():
        status = "✅" if has_key else "❌"
        print(f"{status} {provider}")
    print()

def main():
    """Main entry point."""
    args = parse_args()
    
    # Validate environment
    validate_environment()
    
    try:
        # Initialize and run the orchestrator
        orchestrator = Orchestrator(pdf_paths=args.pdf_paths)
        orchestrator.run()
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Processing interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)