# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
```bash
# Basic usage with direct PDF upload (default)
python main.py <path_to_pdf>

# With text extraction mode
python main.py <path_to_pdf> text_extract

# With web search configuration
python main.py <path_to_pdf> direct_upload all_enabled
python main.py <path_to_pdf> direct_upload all_disabled
python main.py <path_to_pdf> direct_upload solver_only
python main.py <path_to_pdf> direct_upload verifier_only
```

### Testing
```bash
# Run specific tests
python test_network_fallback.py
python test_web_search.py
```

### Installation
```bash
# Install in development mode
pip install -e .

# Or install from PyPI
pip install auto-pset
```

## Architecture Overview

### Core Components

**Main Entry Point (`main.py`)**
- Defines the dual-duo solving strategy with automatic fallback
- Duo 1: Gemini (solver) + Anthropic (verifier)
- Duo 2: Anthropic (solver) + DeepSeek (verifier)
- Handles PDF processing modes and web search configuration

**LLM Provider System (`llm_providers/`)**
- `base.py`: Abstract base class defining the LLM provider interface
- Provider implementations: `anthropic_provider.py`, `gemini_provider.py`, `openai_provider.py`, `deepseek_provider.py`
- Each provider supports PDF uploads, web search (where available), and retry logic
- Providers handle file caching and network fallback

**Configuration (`utils/config.py`)**
- Centralized configuration for all LLM providers
- Manages API keys, model selection, and provider-specific settings
- Handles web search enable/disable for each provider
- Default models: Claude Sonnet 4, Gemini 2.5 Pro, GPT-4o, DeepSeek Reasoner

**PDF Processing (`utils/pdf_extractor.py`)**
- `AdvancedPDFExtractor`: Multi-method text extraction with quality assessment
- Uses pymupdf4llm, pdfplumber, and PyMuPDF with automatic fallback
- Caches extracted text to avoid re-processing
- Optimized for mathematical content and structured documents

**Problem Solving Workflow (`ProblemSetSolver` class)**
- Iterative solve-verify-refine process with configurable rounds
- Supports both direct PDF upload and text extraction modes
- Automatic error handling and retry logic with exponential backoff
- Generates Markdown solutions and converts to LaTeX output

### Key Design Patterns

**Dual-Duo Strategy**: Two independent solver-verifier pairs with automatic fallback if the first duo fails to solve all problems completely.

**Provider Abstraction**: All LLM providers implement the same interface, enabling easy swapping and configuration.

**Multi-Modal Processing**: Supports both direct PDF upload (for providers that support it) and text extraction fallback.

**Caching Strategy**: Files and text extractions are cached to avoid redundant processing and API calls.

**Network Resilience**: Built-in retry logic and network connectivity checking for robust operation.

## Environment Setup

Required environment variables in `.env`:
- `ANTHROPIC_API_KEY`: Claude API key
- `GOOGLE_API_KEY`: Gemini API key  
- `OPENAI_API_KEY`: OpenAI API key
- `DEEPSEEK_API_KEY`: DeepSeek API key
- `PUSHOVER_USER_KEY`: (Optional) For completion notifications
- `PUSHOVER_APP_TOKEN`: (Optional) For completion notifications
- `ANTHROPIC_BASE_URL`: (Optional) Custom Anthropic endpoint

## Output Structure

The application generates outputs in the `outputs/` directory:
- `problem_set.md`: Extracted text from PDF (text extraction mode only)
- `solutions_draft.md`: Current draft of solutions
- `review.md`: Verifier feedback and corrections
- `final_solutions.tex`: Final LaTeX-formatted solutions

## Web Search Configuration

Web search can be configured per provider:
- `default`: Balanced approach (enabled for Anthropic and Gemini)
- `all_enabled`: Enable for all supporting providers
- `all_disabled`: Disable for all providers
- `solver_only`: Enable only for solver providers
- `verifier_only`: Enable only for verifier providers

## Provider-Specific Notes

**Anthropic**: Supports PDF uploads via Files API and web search. Uses system messages for role definition.

**Gemini**: Supports PDF uploads and Google Search grounding. Optimized for mathematical content.

**OpenAI**: Supports PDF uploads but no web search integration yet.

**DeepSeek**: Text-only processing, no PDF upload or web search support.