"""
Summarizer class for creating compact summaries of solver and verifier outputs using DeepSeek.
"""

from typing import Optional
from llm_providers.deepseek_provider import DeepSeekProvider
from utils.config import LLMConfig

class OutputSummarizer:
    """Creates compact summaries of solver and verifier outputs using DeepSeek for memory tracking."""
    
    def __init__(self):
        self.config = LLMConfig()
        self.deepseek = self._init_deepseek()
    
    def _init_deepseek(self) -> Optional[DeepSeekProvider]:
        """Initialize DeepSeek provider for summarization."""
        api_key = self.config.get_api_key(LLMConfig.PROVIDER_DEEPSEEK)
        if not api_key:
            print("Warning: No DeepSeek API key found for summarization")
            return None
        
        provider_config = self.config.get_provider_config(LLMConfig.PROVIDER_DEEPSEEK)
        provider_config['temperature'] = 0.1  # Low temperature for consistent summaries
        
        return DeepSeekProvider(api_key, **provider_config)
    
    def summarize_solver_output(self, solver_output: str, round_num: int, provider_name: str, is_initial: bool = True) -> str:
        """Summarize solver's output using DeepSeek."""
        
        round_type = "initial solution" if is_initial else "refined solution"
        
        if not self.deepseek:
            # Fallback to truncated output if DeepSeek unavailable
            return f"Round {round_num} - {provider_name} ({round_type}): {solver_output[:200]}..."
        
        prompt = f"""Summarize this solver output from {provider_name} in Round {round_num} ({round_type}).

SOLVER OUTPUT:
{solver_output}

Create a concise 3-4 sentence summary covering:
- What problems were addressed and solution approaches used
- Any key findings, results, or code implementations  
- Any challenges, issues, or uncertainties mentioned
- Overall confidence level and completion status

Keep the summary under 200 words and focus on the most important aspects."""

        try:
            summary = self.deepseek.generate(prompt)
            return f"Round {round_num} - {provider_name} ({round_type}):\n{summary.strip()}"
            
        except Exception as e:
            print(f"DeepSeek summarization failed: {e}")
            return f"Round {round_num} - {provider_name} ({round_type}): {solver_output[:200]}..."
    
    def summarize_verifier_output(self, verifier_output: str, round_num: int, provider_name: str) -> str:
        """Summarize verifier's feedback using DeepSeek."""
        
        if not self.deepseek:
            return f"Round {round_num} - {provider_name} (verification): {verifier_output[:200]}..."
        
        prompt = f"""Summarize this verifier feedback from {provider_name} in Round {round_num}.

VERIFIER OUTPUT:
{verifier_output}

Create a concise 2-3 sentence summary covering:
- What errors, issues, or problems were identified
- What corrections or improvements were suggested
- What aspects were verified as correct or acceptable
- Overall assessment and confidence in the verification

Keep the summary under 150 words and focus on actionable feedback."""

        try:
            summary = self.deepseek.generate(prompt)
            return f"Round {round_num} - {provider_name} (verification):\n{summary.strip()}"
            
        except Exception as e:
            print(f"DeepSeek summarization failed: {e}")
            return f"Round {round_num} - {provider_name} (verification): {verifier_output[:200]}..."
    
    def summarize_session_discussions(self, discussions_dir: str, session_summary) -> str:
        """Analyze all discussions from a session and generate lessons for future sessions."""
        
        if not self.deepseek:
            return "DeepSeek not available - cannot generate session lessons."
        
        try:
            from pathlib import Path
            discussions_path = Path(discussions_dir)
            
            # Collect all discussion files from the session
            discussion_files = list(discussions_path.glob("*_discussion.md"))
            
            if not discussion_files:
                return "No discussion files found for lesson generation."
            
            # Simply concatenate all discussions
            all_discussions = ""
            for file_path in discussion_files:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    all_discussions += f"\n# {file_path.stem}\n{content}\n\n"
            
            # Create comprehensive analysis prompt  
            prompt = f"""Analyze this session's problem-solving discussions to generate actionable lessons for future sessions.

SESSION OVERVIEW:
- PDFs Processed: {session_summary.pdfs_processed}
- Success Rate: {session_summary.pdfs_successful}/{session_summary.pdfs_processed}
- Failures: {len(session_summary.failures)}

FAILURE PATTERNS:
{chr(10).join(f"- {failure.failure_type}: {failure.error_message}" for failure in session_summary.failures[:3])}

ALL DISCUSSIONS (INCLUDING INITIAL ANALYSIS AND MODEL SELECTIONS):
{all_discussions}

Generate specific, actionable lessons in these categories:

1. ORCHESTRATOR LESSONS (provider selection, strategy):
2. SOLVER LESSONS (approach, methodology):  
3. VERIFIER LESSONS (review process, accuracy):
4. SYSTEM LESSONS (error patterns, improvements):

Focus on:
- What worked well and should be repeated
- What failed and how to avoid it
- Provider-specific insights
- Subject/difficulty patterns
- Initial analysis accuracy
- Workflow optimizations

Keep each lesson concrete and actionable (1-2 sentences max per lesson)."""

            response = self.deepseek.generate(prompt)
            return response.strip()
            
        except Exception as e:
            print(f"Session discussion summarization failed: {e}")
            return f"Failed to generate session lessons: {str(e)}"