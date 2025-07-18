"""
Orchestrator class that manages the workflow for problem set processing.
It analyzes PDFs, creates execution plans, and configures the parallel processor accordingly.
"""

import os
import json
from typing import TypedDict, List
from dataclasses import dataclass
from enum import Enum

from processors.parallel_processor import ParallelProcessor
from processors.problem_set_solver import PDFMode
from llm_providers.openai_provider import OpenAIProvider
from utils.config import LLMConfig

class Difficulty(Enum):
    """Enum for problem set difficulty levels."""
    HIGH_SCHOOL = "High School"
    UNDERGRADUATE = "Undergraduate"
    GRADUATE = "Graduate"
    RESEARCH = "Research"

class Subject(Enum):
    """Enum for broad subject categories."""
    MATH = "Mathematics"
    PHYSICS = "Physics"
    COMPUTER_SCIENCE = "Computer Science"
    HUMANITIES = "Humanities"
    THEORY = "Theory"
    MIXED = "Mixed"

@dataclass
class PDFAnalysis:
    """Data class for storing PDF analysis results."""
    pdf_path: str
    subject: Subject
    topics: List[str]
    estimated_difficulty: Difficulty
    requires_current_info: bool
    has_code: bool
    has_math: bool
    has_figures: bool
    word_count: int

class ExecutionPlan(TypedDict):
    """Type definition for the execution plan."""
    pdf_path: str
    solver_provider: str
    verifier_provider: str
    rounds: int
    solver_temperature: float
    verifier_temperature: float
    enable_web_search: bool
    analysis_summary: str

class Orchestrator:
    """
    Orchestrates the workflow for processing problem sets.
    Analyzes PDFs, creates execution plans, and manages the parallel processor.
    """
    
    def __init__(self, pdf_paths: List[str]):
        """
        Initialize the Orchestrator.
        
        Args:
            pdf_paths: List of paths to PDF files to process
        """
        self.pdf_paths = pdf_paths
        self._validate_paths()
        
        # Initialize configuration
        self.config = LLMConfig()
        
        # Initialize LLMs for analysis and strategy
        self.strategist_llm = self._init_strategist_llm()
        self.analyst_llm = self._init_analyst_llm()
        
        print(f"🎯 Initialized Orchestrator with {len(pdf_paths)} PDF(s)")
    
    def _validate_paths(self):
        """Validate that all PDF paths exist and are readable."""
        for path in self.pdf_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"PDF not found: {path}")
            if not os.path.isfile(path):
                raise ValueError(f"Not a file: {path}")
            if not os.access(path, os.R_OK):
                raise PermissionError(f"Cannot read file: {path}")
    
    def _init_strategist_llm(self) -> OpenAIProvider:
        """Initialize the strategist LLM (GPT-4) for making execution plans."""
        api_key = self.config.get_api_key(LLMConfig.PROVIDER_OPENAI)
        if not api_key:
            raise ValueError("OpenAI API key not found. The strategist requires GPT-4.")

        provider_config = self.config.get_provider_config(LLMConfig.PROVIDER_OPENAI)
        # Use the highest quality GPT-4 or GPT-4o model for planning
        provider_config['model'] = provider_config.get('model', 'gpt-4o')
        provider_config['temperature'] = 0.2  # Low temperature for deterministic planning

        return OpenAIProvider(api_key=api_key, **provider_config)
    
    def _init_analyst_llm(self) -> OpenAIProvider:
        """Initialize the analyst LLM (GPT-4-Turbo) for quick PDF analysis."""
        api_key = self.config.get_api_key(LLMConfig.PROVIDER_OPENAI)
        if not api_key:
            raise ValueError("OpenAI API key not found. The analyst requires GPT-4-Turbo.")
        
        provider_config = self.config.get_provider_config(LLMConfig.PROVIDER_OPENAI)
        provider_config['model'] = 'gpt-4-0125-preview'  # Use GPT-4-Turbo for fast, accurate analysis
        provider_config['temperature'] = 0.3  # Moderate temperature for analysis
        
        return OpenAIProvider(api_key=api_key, **provider_config)
    
    def _get_analysis_prompt(self, pdf_path: str, error_history: List[str] = None) -> str:
        """Generate the prompt for PDF analysis, with optional error correction."""
        
        correction_prompt = ""
        if error_history:
            correction_prompt = f"""
You have failed on previous attempts. Please correct your mistakes and try again.

PREVIOUS ERRORS:
{"".join([f"- {e}\\n" for e in error_history])}
CRITICAL: Ensure your output is a single, valid JSON object and nothing else.
"""

        return f"""{correction_prompt}Analyze the attached PDF problem set and return a JSON object with the following information:
        
        1. subject: The main subject area (one of: "Mathematics", "Physics", "Computer Science", "Humanities", "Theory", "Mixed")
        2. topics: List of specific topics covered
        3. estimated_difficulty: Difficulty level (one of: "High School", "Undergraduate", "Graduate", "Research")
        4. requires_current_info: Boolean, true if problems require current/recent information
        5. has_code: Boolean, true if any problems involve coding
        6. has_math: Boolean, true if any problems involve mathematical calculations
        7. has_figures: Boolean, true if any problems involve figures, such as a function graph, a table, etc.
        8. word_count: Estimated word count of the problem set
        
        Return ONLY the JSON object, no other text.
        
        Example output format:
        {{
            "subject": "Computer Science",
            "topics": ["Data Structures", "Algorithms", "Time Complexity"],
            "estimated_difficulty": "Undergraduate",
            "requires_current_info": false,
            "has_code": true,
            "has_math": true,
            "has_figures": true,
            "word_count": 2500
        }}
        """
    
    def _get_strategist_prompt(self, analysis: PDFAnalysis, error_history: List[str] = None) -> str:
        """Generate the prompt for creating an execution plan, with optional error correction."""

        correction_prompt = ""
        if error_history:
            correction_prompt = f"""
You have failed on previous attempts. Please correct your mistakes and try again.

PREVIOUS ERRORS:
{"".join([f"- {e}\\n" for e in error_history])}
CRITICAL: Ensure your output is a single, valid JSON object and nothing else.
"""

        return f"""{correction_prompt}You are an expert AI Workflow Strategist. Create an optimal execution plan for solving this problem set.

        Problem Set Analysis:
        ```json
        {{
            "pdf_path": "{analysis.pdf_path}",
            "subject": "{analysis.subject.value}",
            "topics": {json.dumps(analysis.topics)},
            "estimated_difficulty": "{analysis.estimated_difficulty.value}",
            "requires_current_info": {str(analysis.requires_current_info).lower()},
            "has_code": {str(analysis.has_code).lower()},
            "has_math": {str(analysis.has_math).lower()},
            "has_figures": {str(analysis.has_figures).lower()},
            "word_count": {analysis.word_count}
        }}
        ```

        Create an execution plan following these rules:

        1. Provider Duo Selection:
           - If has_figures is False: use ("gemini", "deepseek")
           - For simple tasks, use ("openai", "openai")
           - For all other cases, use ("gemini", "anthropic") or ("anthropic", "gemini"), pick by discretion.

        2. Rounds:
           - High School: 1 round
           - Undergraduate: 2 rounds
           - Graduate/Research: 3 rounds

        3. Temperature:
           - Solver: 0.0-0.5 (lower for math/physics)
           - Verifier: 0.0-0.2 (always low for accuracy)

        4. Web Search:
           - Enable if requires_current_info is true
           - Enable if topics include cutting-edge fields
           - Disable otherwise

        Return ONLY a JSON object in this format:
        {{
            "pdf_path": str,
            "solver_provider": str,
            "verifier_provider": str,
            "rounds": int,
            "solver_temperature": float,
            "verifier_temperature": float,
            "enable_web_search": bool,
            "analysis_summary": str  # One-line explanation
        }}
        """
    
    def _analyze_pdf(self, pdf_path: str) -> PDFAnalysis:
        """
        Analyze a PDF using the analyst LLM, with self-correcting retries.
        """
        print(f"📊 Analyzing PDF: {os.path.basename(pdf_path)}")
        
        max_retries = 3
        error_history = []

        for attempt in range(max_retries):
            prompt = self._get_analysis_prompt(pdf_path, error_history=error_history)
            try:
                print(f"  Attempt {attempt + 1} of {max_retries}...")
                response = self.analyst_llm.generate(prompt, pdf_path=pdf_path)
                data = json.loads(response)
                
                if 'subject' not in data or 'estimated_difficulty' not in data:
                     raise ValueError("Received incomplete analysis data from LLM (missing required fields).")

                return PDFAnalysis(
                    pdf_path=pdf_path,
                    subject=Subject(data['subject']),
                    topics=data.get('topics', ["Unknown"]),
                    estimated_difficulty=Difficulty(data['estimated_difficulty']),
                    requires_current_info=data.get('requires_current_info', False),
                    has_code=data.get('has_code', True),
                    has_math=data.get('has_math', True),
                    has_figures=data.get('has_figures', True),
                    word_count=data.get('word_count', 0)
                )
            except (json.JSONDecodeError, ValueError) as e:
                error_message = f"Error: {e}. Response was: {response[:500]}"
                error_history.append(error_message)
                print(f"  ⚠️ Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    print("  Retrying with error context...")
                else:
                    print(f"  ❌ Failed to analyze {pdf_path} after {max_retries} attempts.")
            except Exception as e:
                print(f"  An unexpected error occurred during PDF analysis: {e}")
                break

        # Fallback to safe defaults
        print("  Falling back to default analysis.")
        return PDFAnalysis(
            pdf_path=pdf_path,
            subject=Subject.MIXED,
            topics=["Unknown"],
            estimated_difficulty=Difficulty.UNDERGRADUATE,
            requires_current_info=True,
            has_code=True,
            has_math=True,
            has_figures=True,
            word_count=0
        )
    
    def _create_execution_plan(self, analysis: PDFAnalysis) -> ExecutionPlan:
        """
        Create an execution plan based on PDF analysis, with self-correcting retries.
        """
        print(f"🎯 Creating execution plan for: {os.path.basename(analysis.pdf_path)}")
        
        max_retries = 3
        error_history = []
        
        for attempt in range(max_retries):
            prompt = self._get_strategist_prompt(analysis, error_history=error_history)
            try:
                print(f"  Attempt {attempt + 1} of {max_retries}...")
                response = self.strategist_llm.generate(prompt)
                safe_response = response.strip()
                if safe_response.lower().startswith('```'):
                    # find first newline after the opening fence and last fence
                    safe_response = safe_response.split('\n', 1)[-1]
                    if safe_response.endswith('```'):
                        safe_response = safe_response[:-3]
                plan = json.loads(safe_response)

                if not plan or 'pdf_path' not in plan:
                    raise ValueError("Received incomplete or empty plan from LLM.")
                return plan
            except (json.JSONDecodeError, ValueError) as e:
                error_message = f"Error: {e}. Response was: {response[:500]}"
                error_history.append(error_message)
                print(f"  ⚠️ Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    print("  Retrying with error context...")
                else:
                    print(f"  ❌ Failed to create plan for {analysis.pdf_path} after {max_retries} attempts.")
            except Exception as e:
                print(f"  An unexpected error occurred during plan creation: {e}")
                break

        # Fallback to safe defaults if all retries fail
        print("  Falling back to default execution plan.")
        return {
            "pdf_path": analysis.pdf_path,
            "solver_provider": "anthropic",
            "verifier_provider": "gemini",
            "rounds": 2,
            "solver_temperature": 0.1,
            "verifier_temperature": 0.0,
            "enable_web_search": True,
            "analysis_summary": "Using safe defaults due to a planning failure."
        }
    
    def run(self) -> None:
        """
        Run the orchestrated workflow:
        1. Analyze all PDFs
        2. Create execution plans
        3. Configure and run the parallel processor
        """
        print("\n🚀 Starting Orchestrated Workflow")
        print("=" * 50)
        
        # Step 1: Analyze all PDFs
        analyses = []
        for pdf_path in self.pdf_paths:
            analysis = self._analyze_pdf(pdf_path)
            analyses.append(analysis)
            print(f"✓ Analyzed: {os.path.basename(pdf_path)}")
            print(f"  Subject: {analysis.subject.value}")
            print(f"  Difficulty: {analysis.estimated_difficulty.value}")
            print()
        
        # Step 2: Create execution plans
        execution_plans = []
        for analysis in analyses:
            plan = self._create_execution_plan(analysis)
            execution_plans.append(plan)
            print(f"✓ Plan created for: {os.path.basename(plan['pdf_path'])}")
            print(f"  Duo: {plan['solver_provider']}/{plan['verifier_provider']}")
            print(f"  Rounds: {plan['rounds']}")
            print(f"  Web Search: {'Enabled' if plan['enable_web_search'] else 'Disabled'}")
            print()
        
        # Step 3: Configure provider duos for parallel processor
        provider_duos = [(plan['solver_provider'], plan['verifier_provider']) 
                        for plan in execution_plans]
        
        # Step 4: Initialize and run parallel processor
        print("\n🏃 Executing Parallel Processing")
        print("=" * 50)
        
        processor = ParallelProcessor(
            pdf_paths=self.pdf_paths,
            provider_duos=provider_duos,
            pdf_mode=PDFMode.DIRECT_UPLOAD
        )
        
        # Run the processor
        processor.run()
        
        print("\n✅ Orchestrated Workflow Complete!")
        print("=" * 50)
        
        # Print final summary
        print("\n📊 Execution Summary:")
        for plan in execution_plans:
            print(f"\n• {os.path.basename(plan['pdf_path'])}:")
            print(f"  {plan['analysis_summary']}") 