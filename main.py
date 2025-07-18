import sys
import os
import requests
from dotenv import load_dotenv

# Add project root to the Python path to allow absolute imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from utils.config import LLMConfig
from processors.problem_set_solver import PDFMode
from processors.parallel_processor import ParallelProcessor

# Load environment variables from .env file
load_dotenv()

# --- MCP Client Cleanup ---
# Note: Cleanup is now handled by the processor that manages the client lifecycle.
try:
    from mcp_server.client import disconnect_mcp_client
    MCP_CLEANUP_AVAILABLE = True
except ImportError:
    MCP_CLEANUP_AVAILABLE = False
    disconnect_mcp_client = None

def main():
    """
    Main entry point for the Auto-PSET Solver.
    """
    if len(sys.argv) < 2:
        print("Usage: python main.py <pdf_path_1> [pdf_path_2] ...")
        sys.exit(1)

    # --- Argument Parsing (Simplified) ---
    pdf_paths = [arg for arg in sys.argv[1:] if arg.lower().endswith('.pdf')]
    
    # Verify all provided PDF paths
    for path in pdf_paths:
        if not os.path.exists(path):
            print(f"❌ Error: PDF file not found at '{path}'")
            sys.exit(1)

    if not pdf_paths:
        print("❌ Error: No valid PDF files were provided.")
        sys.exit(1)

    print(f"📚 PDF files to process: {pdf_paths}")
    print("-" * 50)

    # --- Hardcoded Configuration ---
    pdf_mode = PDFMode.DIRECT_UPLOAD
    config = LLMConfig()
    # Default: Enable web search for verifiers, disable for solvers
    config.set_web_search_enabled(LLMConfig.PROVIDER_ANTHROPIC, True)
    config.set_web_search_enabled(LLMConfig.PROVIDER_GEMINI, True)
    config.set_web_search_enabled(LLMConfig.PROVIDER_OPENAI, False)
    config.set_web_search_enabled(LLMConfig.PROVIDER_DEEPSEEK, False)
        
    print("\n📊 Using default web search configuration:")
    web_search_status = config.get_web_search_status()
    for provider, enabled in web_search_status.items():
        print(f"  {provider}: {'✅ ENABLED' if enabled else '❌ DISABLED'}")
    print("-" * 50)

    # --- Provider Duo Configuration ---
    solver_provider = LLMConfig.PROVIDER_GEMINI
    verifier_provider = LLMConfig.PROVIDER_ANTHROPIC
    
    print(f"🤖 Using Solver: {solver_provider}")
    print(f"🕵️ Using Verifier: {verifier_provider}")
    print("-" * 50)

    # --- Start Parallel Processing ---
    provider_duos = [
        ("gemini", "anthropic"),     
        ("anthropic", "gemini")
    ]
    processor = ParallelProcessor(
        pdf_paths=pdf_paths,
        provider_duos=provider_duos,
        pdf_mode=pdf_mode
    )
    
    processor.run()
    
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ A critical error occurred in the main workflow: {e}")
    finally:
        if MCP_CLEANUP_AVAILABLE and disconnect_mcp_client:
            disconnect_mcp_client()
            print("🗑️ Cleaned up MCP client connection.")
        print("\nAll tasks complete.")