"""
Orchestrator class that manages the workflow for problem set processing.
It analyzes PDFs, creates execution plans, and configures the parallel processor accordingly.
"""

import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypedDict, List
from dataclasses import dataclass
from enum import Enum

from processors.parallel_processor import ParallelProcessor
from processors.problem_set_solver import PDFMode
from llm_providers.openai_provider import OpenAIProvider
from utils.config import LLMConfig
from utils.memory_manager import MemoryManager

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
        
        # Initialize shared LLM for both analysis and strategy to reduce overhead
        self.shared_llm = self._init_shared_llm()
        
        # Initialize memory manager if enabled in config
        self.memory = MemoryManager() if self.config.is_memory_enabled() else None
        
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
    
    def _init_shared_llm(self) -> OpenAIProvider:
        """Initialize shared LLM for both analysis and strategy to reduce overhead."""
        api_key = self.config.get_api_key(LLMConfig.PROVIDER_OPENAI)
        if not api_key:
            raise ValueError("OpenAI API key not found. The orchestrator requires GPT-4.")

        provider_config = self.config.get_provider_config(LLMConfig.PROVIDER_OPENAI)
        provider_config['model'] = provider_config.get('model', 'gpt-4o')
        provider_config['temperature'] = 0.1  # Balanced temperature for both tasks

        return OpenAIProvider(api_key=api_key, **provider_config)
    
    def _get_analysis_prompt(self, pdf_path: str, error_history: List[str] = None) -> str:
        """Generate the prompt for PDF analysis, with optional error correction."""
        
        # Get session memory if available
        session_context = ""
        if self.memory:
            recent_lessons = self.memory.get_recent_session_lessons(2)
            if recent_lessons:
                # Extract only orchestrator-relevant lessons
                lesson_lines = recent_lessons.split('\n')
                orchestrator_lessons = []
                capture = False
                for line in lesson_lines:
                    if "ORCHESTRATOR LESSONS" in line:
                        capture = True
                        continue
                    elif capture and ("SOLVER LESSONS" in line or "VERIFIER LESSONS" in line or "SYSTEM LESSONS" in line):
                        break
                    elif capture and line.strip():
                        orchestrator_lessons.append(line.strip())
                
                if orchestrator_lessons:
                    session_context = f"""
### Previous Session Insights:
{chr(10).join(orchestrator_lessons[:3])}

"""
        
        correction_prompt = ""
        if error_history:
            correction_prompt = f"""
You have failed on previous attempts. Please correct your mistakes and try again.

PREVIOUS ERRORS:
{"".join([f"- {e}\\n" for e in error_history])}

CRITICAL INSTRUCTIONS:
- Return ONLY a valid JSON object, no other text, explanations, or markdown
- Do not wrap the JSON in ```json``` code blocks
- Ensure all required fields are included
- Double-check your JSON syntax before responding
"""

        return f"""{correction_prompt}{session_context}Analyze the attached PDF problem set and return a JSON object with the following information:
        
        1. subject: The main subject area (one of: "Mathematics", "Physics", "Computer Science", "Humanities", "Theory", "Mixed")
        2. topics: List of specific topics covered
        3. estimated_difficulty: Difficulty level (one of: "High School", "Undergraduate", "Graduate", "Research")
        4. requires_current_info: Boolean, true if problems require current/recent information
        5. has_code: Boolean, true if any problems involve coding
        6. has_math: Boolean, true if any problems involve mathematical calculations
        7. has_figures: Boolean, true if any problems involve figures, such as a function graph, a table, etc.
        8. word_count: Estimated word count of the problem set
        
        CRITICAL: Return ONLY a valid JSON object with no additional text, markdown, or explanations.
        
        Required output format (copy exactly):
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

        # Get session memory if available
        session_context = ""
        if self.memory:
            recent_lessons = self.memory.get_recent_session_lessons(2)
            if recent_lessons:
                # Extract orchestrator-relevant lessons for strategy
                lesson_lines = recent_lessons.split('\n')
                orchestrator_lessons = []
                capture = False
                for line in lesson_lines:
                    if "ORCHESTRATOR LESSONS" in line:
                        capture = True
                        continue
                    elif capture and ("SOLVER LESSONS" in line or "VERIFIER LESSONS" in line or "SYSTEM LESSONS" in line):
                        break
                    elif capture and line.strip():
                        orchestrator_lessons.append(line.strip())
                
                if orchestrator_lessons:
                    session_context = f"""
### Lessons from Previous Sessions:
{chr(10).join(orchestrator_lessons[:4])}

Apply these insights when making provider and strategy decisions.

"""

        correction_prompt = ""
        if error_history:
            correction_prompt = f"""
You have failed on previous attempts. Please correct your mistakes and try again.

PREVIOUS ERRORS:
{"".join([f"- {e}\\n" for e in error_history])}

CRITICAL INSTRUCTIONS:
- Return ONLY a valid JSON object, no other text, explanations, or markdown
- Do not wrap the JSON in ```json``` code blocks
- Ensure all required fields are included
- Double-check your JSON syntax before responding
"""

        return f"""{correction_prompt}{session_context}You are an expert AI Workflow Strategist. Create an optimal execution plan for solving this problem set.

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

        CRITICAL: Return ONLY a valid JSON object with no additional text, markdown, or explanations.
        
        Required output format (copy exactly):
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
                response = self.shared_llm.generate(prompt, pdf_path=pdf_path)
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
                response = self.shared_llm.generate(prompt)
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
    
    def _process_pdf(self, pdf_path: str) -> tuple[PDFAnalysis, ExecutionPlan]:
        """Process a single PDF: analyze and create execution plan."""
        analysis = self._analyze_pdf(pdf_path)
        plan = self._create_execution_plan(analysis)
        return analysis, plan
    
    def run(self) -> None:
        """
        Run the orchestrated workflow with parallel processing:
        1. Analyze all PDFs in parallel
        2. Create execution plans in parallel
        3. Configure and run the parallel processor
        """
        print("\n🚀 Starting Orchestrated Workflow")
        print("=" * 50)
        
        # Start memory session if memory is enabled
        if self.memory:
            session_id = self.memory.start_session()
            print(f"📝 Memory session started: {session_id}")
        else:
            print("🧠 Memory system disabled")
        
        analyses = []
        execution_plans = []
        
        print("\n📊 Processing PDFs in parallel...")
        
        # Process all PDFs in parallel for better performance
        with ThreadPoolExecutor(max_workers=min(4, len(self.pdf_paths))) as executor:
            # Submit all PDF processing tasks
            future_to_pdf = {executor.submit(self._process_pdf, pdf_path): pdf_path 
                           for pdf_path in self.pdf_paths}
            
            # Collect results as they complete
            for future in as_completed(future_to_pdf):
                pdf_path = future_to_pdf[future]
                try:
                    analysis, plan = future.result()
                    analyses.append(analysis)
                    execution_plans.append(plan)
                    
                    # Start discussion tracking for this PDF if memory is enabled
                    if self.memory:
                        analysis_dict = {
                            'subject': analysis.subject.value,
                            'estimated_difficulty': analysis.estimated_difficulty.value,
                            'topics': analysis.topics,
                            'has_math': analysis.has_math,
                            'has_code': analysis.has_code,
                            'has_figures': analysis.has_figures,
                            'enable_web_search': plan['enable_web_search']
                        }
                        self.memory.start_pdf_discussion(pdf_path, analysis_dict)
                        
                        # Record successful processing
                        self.memory.record_pdf_result(pdf_path, True)
                    
                    print(f"✓ Processed: {os.path.basename(pdf_path)}")
                    print(f"  Subject: {analysis.subject.value}")
                    print(f"  Difficulty: {analysis.estimated_difficulty.value}")
                    print(f"  Duo: {plan['solver_provider']}/{plan['verifier_provider']}")
                    print(f"  Rounds: {plan['rounds']}")
                    print(f"  Web Search: {'Enabled' if plan['enable_web_search'] else 'Disabled'}")
                    print()
                except Exception as e:
                    print(f"❌ Failed to process {pdf_path}: {e}")
                    
                    # Record failed processing if memory is enabled
                    if self.memory:
                        self.memory.record_pdf_result(pdf_path, False, str(e), "processing")
                    
                    # Use fallback for failed processing
                    fallback_analysis = PDFAnalysis(
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
                    fallback_plan = {
                        "pdf_path": pdf_path,
                        "solver_provider": "anthropic",
                        "verifier_provider": "gemini",
                        "rounds": 2,
                        "solver_temperature": 0.1,
                        "verifier_temperature": 0.0,
                        "enable_web_search": True,
                        "analysis_summary": "Using fallback due to processing failure."
                    }
                    analyses.append(fallback_analysis)
                    execution_plans.append(fallback_plan)
        
        # Step 3: Configure provider duos for parallel processor
        provider_duos = [(plan['solver_provider'], plan['verifier_provider']) 
                        for plan in execution_plans]
        
        # Step 4: Initialize and run parallel processor
        print("\n🏃 Executing Parallel Processing")
        print("=" * 50)
        
        processor = ParallelProcessor(
            pdf_paths=self.pdf_paths,
            provider_duos=provider_duos,
            pdf_mode=PDFMode.DIRECT_UPLOAD,
            memory_manager=self.memory
        )
        
        # Run the processor
        processor.run()
        
        print("\n✅ Orchestrated Workflow Complete!")
        print("=" * 50)
        
        # End memory session and save learnings if memory is enabled
        if self.memory:
            self.memory.end_session()
        
        # Print final summary
        print("\n📊 Execution Summary:")
        for plan in execution_plans:
            print(f"\n• {os.path.basename(plan['pdf_path'])}:")
            print(f"  {plan['analysis_summary']}") 