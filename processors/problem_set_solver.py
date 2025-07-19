import os
from typing import Optional, Dict, Any, Union, List
from enum import Enum
import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from httpx import ConnectError
from anthropic._exceptions import OverloadedError
from google.genai.errors import ServerError as GeminiServerError

from utils.config import LLMConfig
from utils.pdf_extractor import AdvancedPDFExtractor
from processors.summarizer import OutputSummarizer
from llm_providers.openai_provider import OpenAIProvider
from llm_providers.anthropic_provider import AnthropicProvider
from llm_providers.gemini_provider import GeminiProvider
from llm_providers.deepseek_provider import DeepSeekProvider
from llm_providers.base import BaseLLMProvider

# Try to import the new LatexStyler
try:
    from processors.latex_styler import LatexStyler
    STYLING_AVAILABLE = True
except ImportError:
    STYLING_AVAILABLE = False
    LatexStyler = None

# Try to import MCP client for cleanup
try:
    from mcp_server.client import disconnect_mcp_client
    MCP_CLEANUP_AVAILABLE = True
except ImportError:
    MCP_CLEANUP_AVAILABLE = False
    disconnect_mcp_client = None

class PDFMode(Enum):
    """Enum for PDF processing modes."""
    DIRECT_UPLOAD = "direct_upload"  # Send PDF directly to LLM
    TEXT_EXTRACT = "text_extract"    # Extract text first

class ProblemSetSolver:
    """Main class for processing problem sets with multiple LLM providers."""
    
    def __init__(self, 
                 solver_provider: str = "gemini", 
                 verifier_provider: str = "anthropic",
                 pdf_mode: PDFMode = PDFMode.DIRECT_UPLOAD,
                 rounds: int = 2,
                 memory_manager = None):
        self.config = LLMConfig()
        self.pdf_mode = pdf_mode
        self.rounds = rounds
        self.solve_complete = False
        self.memory_manager = memory_manager if memory_manager and self.config.is_memory_enabled() else None
        self.solver_provider_name = solver_provider
        self.verifier_provider_name = verifier_provider
        self.summarizer = OutputSummarizer() if self.memory_manager else None
        self.pdf_extractor = AdvancedPDFExtractor()
        
        if pdf_mode == PDFMode.TEXT_EXTRACT:
            self.pdf_extractor = AdvancedPDFExtractor()
        
        # Initialize providers
        self.solver = self._init_provider(solver_provider)
        self.verifier = self._init_provider(verifier_provider)
        
        if not self.solver or not self.verifier:
            available = self.config.get_available_providers()
            print("\nAvailable providers:", {k: v for k, v in available.items() if v})
            raise ValueError("Could not initialize required providers. Check your API keys.")
    
    def _init_provider(self, provider_type: str) -> Optional[BaseLLMProvider]:
        """Initialize a provider if API key is available."""
        api_key = self.config.get_api_key(provider_type)
        if not api_key:
            print(f"Warning: No API key found for {provider_type}")
            return None
            
        provider_config = self.config.get_provider_config(provider_type)
        
        # Add system message for Claude and custom base URL
        if provider_type == LLMConfig.PROVIDER_ANTHROPIC:
            provider_config['system'] = "You are an expert academic reviewer with an eye for detail."
        
        providers = {
            LLMConfig.PROVIDER_OPENAI: OpenAIProvider,
            LLMConfig.PROVIDER_ANTHROPIC: AnthropicProvider,
            LLMConfig.PROVIDER_GEMINI: GeminiProvider,
            LLMConfig.PROVIDER_DEEPSEEK: DeepSeekProvider
        }
        
        provider_class = providers.get(provider_type)
        if provider_class:
            # Extract enable_web_search from provider_config for explicit passing
            enable_web_search = provider_config.pop('enable_web_search', True)
            return provider_class(api_key, enable_web_search=enable_web_search, **provider_config)
        return None

    def _get_retry_decorator(self):
        """Creates a tenacity retry decorator for handling transient API errors."""
        return retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=4, max=20),
            retry=retry_if_exception_type((ConnectError, OverloadedError, anthropic.RateLimitError, GeminiServerError)),
        )

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extracts text content from a PDF file using advanced extraction methods."""
        print("📄 Extracting text from PDF...")
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"The file {pdf_path} was not found.")
        return self.pdf_extractor.extract_text(pdf_path)

    def generate_initial_solutions(self, input_data: Union[str, List[str]], is_pdf_path: bool = False, hints: Optional[str] = None) -> str:
        """Prompts the solver LLM to generate solutions in Markdown."""
        @self._get_retry_decorator()
        def _generate():
            print("🤖 Prompting Solver LLM to generate solutions...")
            print(f"Using solver: {self.solver.get_name()}")

            # Get session memory for solver lessons if available
            solver_context = ""
            if self.memory_manager:
                recent_lessons = self.memory_manager.get_recent_session_lessons(2)
                if recent_lessons:
                    # Extract solver-relevant lessons
                    lesson_lines = recent_lessons.split('\n')
                    solver_lessons = []
                    capture = False
                    for line in lesson_lines:
                        if "SOLVER LESSONS" in line:
                            capture = True
                            continue
                        elif capture and ("VERIFIER LESSONS" in line or "SYSTEM LESSONS" in line or "ORCHESTRATOR LESSONS" in line):
                            break
                        elif capture and line.strip():
                            solver_lessons.append(line.strip())
                    
                    if solver_lessons:
                        solver_context = f"""
### Previous Session Learnings:
{chr(10).join(solver_lessons[:3])}

Apply these insights to your problem-solving approach.

"""

            solver_supports_pdf = self.solver.supports_pdf_upload
            pdf_path_arg = None
            
            if is_pdf_path and solver_supports_pdf:
                prompt = f"""
                You are an expert mathematician, programmer, scientist, and problem solver.
                {solver_context}
                INSTRUCTIONS:
                1. Read through the entire PDF to identify ALL problems/exercises, including both mathematical and coding problems.
                2. For EACH problem found, provide a complete solution
                3. For coding problems, implement working code with proper testing using bash tools if available.
                
                FORMAT: Start each solution with "## Exercise X:" where X is the problem number.
                
                Please solve ALL problems from the provided PDF with detailed, step-by-step explanations for each problem.
                Make sure to identify each individual problem/exercise and provide a complete solution for every single one.
                Output the solutions in well-formatted Markdown.
                """
                pdf_path_arg = str(input_data)
            else:
                # If solver doesn't support PDF or if input is already text
                if is_pdf_path and not solver_supports_pdf:
                    print(f"🔄 Solver {self.solver.get_name()} doesn't support PDF upload. Extracting text...")
                    final_input_data = self.extract_text_from_pdf(str(input_data))
                else:
                    final_input_data = input_data

                prompt = f"""
                You are an expert mathematician, scientist, and problem solver.
                {solver_context}
                Please solve ALL problems from the following problem set with detailed, step-by-step explanations for each problem.
                Make sure to identify each individual problem/exercise and provide a complete solution for every single one.
                Include both mathematical problems and coding problems if present.
                FORMAT: Start each solution with "## Exercise X:" where X is the problem number.
                
                Please solve ALL problems from the provided PDF with detailed, step-by-step explanations for each problem.
                Make sure to identify each individual problem/exercise and provide a complete solution for every single one.
                Output the solutions in well-formatted Markdown.
                
                ## Problem Set:
                {final_input_data}
                """
            
            if hints:
                prompt += f"""
                
                ### Starter Hints
                Please use these hints as a starting point for your solutions:
                {hints}
                """
            
            # Add detailed tool instructions only if the provider supports
            # code execution or MCP tools. Otherwise, keep the prompt simpler.
            if (self.solver.supports_code_execution or self.solver.supports_mcp) and self.solver.get_name() == LLMConfig.PROVIDER_ANTHROPIC:
                prompt += """
                ## Problem-Specific Approaches

                ### For Mathematical Problems:
                - **Only use MCP tools for calculus, symbolic algebra, matrices, roots of high-degree polynomials, etc.**
                - **Perform ordinary arithmetic (add/subtract/multiply/divide/simple roots, decimal evaluation) yourself.**
                - For complex calculations (e.g., integrals, derivatives, matrices), use the `mcp_generic_math` tools.
                - **CRITICAL: You have a MAX_TOOL_TURNS limit of 5. Excessive tool calls will invalidate your answer.**

                ### For Coding Problems:
                **EFFICIENT WORKFLOW:**
                1. **Plan First**: Analyze the problem requirements briefly
                2. **Code Efficiently**: Write clean, working code using bash tools if available
                3. **Test Once**: Verify the solution works correctly with unit testing scripts
                4. **Present Cleanly**: Include only the final, working solution

                **CODING STANDARDS:**
                - Include only essential functionality (avoid over-engineering)
                - Present the final solution without showing debugging steps

                ## Output Format Standards

                ### Problem Structure:
                ```markdown
                ## Exercise [Number]: [Brief Problem Description]

                ### Solution:
                [Concise explanation of approach]

                [Working code or mathematical solution]

                ### Answer: [Final result prominently displayed]
                ```

                ## Quality Standards

                ### DO:
                - Use tools for verification, not for showing work
                - Present clean, final solutions
                - Test code before presenting it

                ### DON'T:
                - Show debugging steps or failed attempts
                - Include raw tool output in final response
                - Over-engineer simple solutions
                - Provide multiple variations unless specifically requested
                - Include excessive explanatory text for straightforward problems

                ## Sandbox Environment Limitations
                
                **Available Commands in Sandbox:**
                - `python3` - Python interpreter
                - `echo`, `cat`, `ls`, `grep`, `sort`, `head`, `tail`, `wc` - Basic utilities
                - File operations: `>`, `>>`, `|`, `&&`

                **NOT Available in Sandbox:**
                - `pdflatex`, `latex` - LaTeX compilation (generate LaTeX code only, don't compile)
                - `apt-get`, `yum`, `brew` - Package managers
                - `sudo`, `docker` - System administration tools

                ## Tool Usage Efficiency
                
                **MINIMIZE TOOL CALLS:**
                - Plan your approach before using tools
                - Combine multiple operations in single bash commands where possible
                - Use tools strategically for verification, not exploration.
                - **For math, only use tools for complex calculations - NOT simple arithmetic.**
                - **Remember: MAX_TOOL_TURNS limit is 5. Excessive tool calls will invalidate your answer.**

                **CRITICAL: Solve ALL problems in the PDF efficiently. Quality over quantity.**
                **REMEMBER: Tools are for YOUR internal use only. Present clean, final solutions.**
                **IMPORTANT: Don't attempt to compile LaTeX or install packages in sandbox.**
                """
            
            try:
                print(f"Processing {'PDF' if pdf_path_arg else 'text'} input...")
                if pdf_path_arg:
                    response = self.solver.generate(prompt, pdf_path=pdf_path_arg)
                else:
                    response = self.solver.generate(prompt)
                print(f"Response length: {len(response)} characters")
                
                if not response or not response.strip():
                    print("⚠️ Warning: Solver returned empty response")
                    return "Error: No solution generated by the solver LLM."
                
                return response
            except Exception as e:
                print(f"❌ Error in generate_solutions: {e}")
                raise
        return _generate()

    def verify_solutions(self, original_input: Union[str, List[str]], solution_text: str, is_pdf_path: bool = False, hints: Optional[str] = None, pdf_path: str = None) -> str:
        """Asks the verifier LLM to check the solutions for mistakes."""
        @self._get_retry_decorator()
        def _verify():
            print("🕵️ Asking Verifier LLM to check the solutions...")
            print(f"Using verifier: {self.verifier.get_name()}")
            
            # Get session memory for verifier lessons if available
            verifier_context = ""
            if self.memory_manager:
                recent_lessons = self.memory_manager.get_recent_session_lessons(2)
                if recent_lessons:
                    # Extract verifier-relevant lessons
                    lesson_lines = recent_lessons.split('\n')
                    verifier_lessons = []
                    capture = False
                    for line in lesson_lines:
                        if "VERIFIER LESSONS" in line:
                            capture = True
                            continue
                        elif capture and ("SYSTEM LESSONS" in line or "ORCHESTRATOR LESSONS" in line or "SOLVER LESSONS" in line):
                            break
                        elif capture and line.strip():
                            verifier_lessons.append(line.strip())
                    
                    if verifier_lessons:
                        verifier_context = f"""
### Previous Session Verifier Insights:
{chr(10).join(verifier_lessons[:3])}

Apply these insights to your verification approach.

"""
            
            # Get previous discussion context if available  
            context = ""
            if self.memory_manager and pdf_path:
                full_context = self.memory_manager.get_discussion_context(pdf_path)
                if full_context:
                    # Extract just the last few key points for slim context
                    lines = full_context.strip().split('\n')
                    recent_lines = [line for line in lines[-8:] if line.strip() and not line.startswith('#')]
                    if recent_lines:
                        context = f"\n### Previous Context:\n{chr(10).join(recent_lines[-4:])}\n"
            
            # Check if verifier supports PDF uploads directly via the provider property
            verifier_supports_pdf = self.verifier.supports_pdf_upload
            pdf_path_arg = None
            
            if is_pdf_path and verifier_supports_pdf:
                prompt = f"""
                The attached PDF contains a problem set. Below are my drafted solutions.
                Please review them against the original problems in the PDF and indicate any errors.
                {verifier_context}{context}
                ### Proposed Solutions
                {solution_text}
                """
                pdf_path_arg = str(original_input)
            else:
                if is_pdf_path and not verifier_supports_pdf:
                    # Extract text from PDF for verifiers that don't support direct upload
                    print(f"🔄 Verifier {self.verifier.get_name()} doesn't support PDF upload. Extracting text...")
                    if not hasattr(self, 'pdf_extractor'):
                        from utils.pdf_extractor import AdvancedPDFExtractor
                        self.pdf_extractor = AdvancedPDFExtractor()
                    
                    final_input_data = self.pdf_extractor.extract_text(str(original_input))
                else:
                    final_input_data = original_input

                prompt = f"""
                ### Original Problem Set
                {final_input_data}
                {verifier_context}{context}
                ### Proposed Solutions
                {solution_text}
                """
                
            if hints:
                prompt += f"""
                
                ### Starter Hints
                Please incorporate these hints into your review:
                {hints}
                """
            if (self.verifier.supports_code_execution or self.verifier.supports_mcp) and self.verifier.get_name() == LLMConfig.PROVIDER_ANTHROPIC:
                prompt += f"""
                ## Verification Approach

                ### For Mathematical Problems:
                - **Only use MCP tools for calculus, symbolic algebra, matrices, roots of high-degree polynomials, etc.**
                - **Perform ordinary arithmetic (add/subtract/multiply/divide/simple roots, decimal evaluation) yourself.**
                - Use MCP math tools to verify complex calculations only
                - Check mathematical reasoning and logic
                - Verify that final answers are correct
                - Check for proper mathematical notation and formatting
                - **CRITICAL: You have a MAX_TOOL_TURNS limit of 5. Excessive tool calls will invalidate your answer.**

                ### For Coding Problems:
                **VERIFICATION WORKFLOW:**
                1. **Code Analysis**: Review the logic and algorithm
                2. **Execution Testing**: Use bash tools to test the code
                3. **Edge Case Testing**: Verify with different inputs
                4. **Quality Assessment**: Check code style and efficiency

                ## Review Output Format

                ### For Correct Solutions:
                ```markdown
                ## Exercise [Number]: ✅ VERIFIED CORRECT
                ```

                ### For Incorrect Solutions:
                ```markdown
                ## Exercise [Number]: ❌ ISSUES FOUND

                ### Issues Identified:
                1. **[Issue Type]**: [Specific problem description]
                - **Fix Action**: [Specific correction needed]

                ### Verification Evidence:
                [Tool output or calculation]

                **INSTRUCTIONS:**
                1. Systematically verify all solutions.
                2. Use tools to check code and calculations.
                3. Give concise, actionable feedback for any issues.

                **OUTPUT:**
                - If all correct: put "final answer to all problems: no mistakes found." at the end of your response.
                - If issues: List each issue with fix action and supporting evidence.

                Begin your review.
                """
            else:
                prompt += """
                ## Review Output Format

                ### For Correct Solutions:
                ```markdown
                ## Exercise [Number]: ✅ VERIFIED CORRECT
                ```

                ### For Incorrect Solutions:
                ```markdown
                ## Exercise [Number]: ❌ ISSUES FOUND

                ### Issues Identified:
                1. **[Issue Type]**: [Specific problem description]
                - **Fix Action**: [Specific correction needed]

                ### Verification Evidence:
                [Tool output or calculation]

                **INSTRUCTIONS:**
                1. Systematically verify all solutions.
                2. Give concise, actionable feedback for any issues.

                **OUTPUT:**
                - If all correct: put "final answer to all problems: no mistakes found." at the end of your response. 
                - as long as all final answers are correct, you can say "final answer to all problems: no mistakes found."
                - If issues: List each issue with fix action and supporting evidence.

                Begin your review.
                """
            
            try:
                print(f"Verifying solutions...")
                if pdf_path_arg:
                    response = self.verifier.generate(prompt, pdf_path=pdf_path_arg)
                else:
                    response = self.verifier.generate(prompt)
                print(f"Verification response length: {len(response)} characters")
                
                if not response or not response.strip():
                    print("⚠️ Warning: Verifier returned empty response")
                    return "Error: No verification response generated."
                
                return response
            except Exception as e:
                print(f"❌ Error in verify_solutions: {e}")
                raise
        return _verify()

    def apply_fixes_and_regenerate(self, original_input: Union[str, List[str]], solution_text: str, fix_instructions: str, is_pdf_path: bool = False, pdf_path: str = None) -> str:
        """Prompts the solver LLM to apply fixes and generate a new round of solutions."""
        @self._get_retry_decorator()
        def _regenerate():
            print("⚙️ Applying fixes and re-solving...")

            # Get previous discussion context if available
            context = ""
            if self.memory_manager and pdf_path:
                context = self.memory_manager.get_discussion_context(pdf_path)

            solver_supports_pdf = self.solver.supports_pdf_upload
            pdf_path_arg = None
            
            if is_pdf_path and solver_supports_pdf:
                prompt = """
                You are an expert mathematician, scientist, and problem solver. The attached PDF contains a problem set.
                You previously generated solutions for this problem set, which are included below along with suggestions for corrections from a reviewer.
                Your task is to re-solve the problems from the PDF, incorporating the feedback to produce a new, corrected set of solutions.
                Take their suggestions and fix actions with a grain of salt, they might be wrong or right, think hard for yourself whenever their suggestions disagree with your previous solution.
                Output the final, corrected solutions in well-formatted Markdown.
                """
                pdf_path_arg = str(original_input)
            else:
                if is_pdf_path and not solver_supports_pdf:
                    print(f"🔄 Solver {self.solver.get_name()} doesn't support PDF upload. Extracting text...")
                    final_input_data = self.extract_text_from_pdf(str(original_input))
                else:
                    final_input_data = original_input
                
                prompt = f"""
                You are an expert mathematician, scientist, and problem solver. 
                You have provided solutions to the problem set before, and a peer expert has provided correction suggestions. 
                Please solve the problems from the provided PDF with detailed, step-by-step explanations for each problem, incorporating the suggestions.
                Take their suggestions and fix actions with a grain of salt, they might be wrong or right, think hard for yourself whenever their suggestions disagree with your previous solution.
                Output the solutions in well-formatted Markdown.

                ### Original Problem Set
                {final_input_data}
                """
                
            prompt += f"""
            {context}
            ### Original Solutions (in Markdown)
            {solution_text}

            ### Fix suggestions
            {fix_instructions}

            Don't omit details, preserve the correct step-by-step process in the original solutions and correct the wrong ones.
            """
            if pdf_path_arg:
                return self.solver.generate(prompt, pdf_path=pdf_path_arg)
            return self.solver.generate(prompt)
        return _regenerate()

    def apply_fixes_and_generate_latex(self, original_input: Union[str, List[str]], solution_text: str, fix_instructions: Optional[str] = None, format_instructions: Optional[str] = None, is_pdf_path: bool = False) -> str:
        """Prompts the solver LLM to apply fixes and generate final LaTeX output."""
        @self._get_retry_decorator()
        def _generate_latex():
            print("⚙️ Applying fixes and generating final LaTeX output...")
            
            solver_supports_pdf = self.solver.supports_pdf_upload
            pdf_path_arg = None

            if is_pdf_path and solver_supports_pdf:
                prompt = """
                You are a LaTeX expert. The attached PDF contains the original problem set.
                Below, you will find a set of solutions previously generated in Markdown, along with review comments.
                Your task is to apply the corrections and generate a final, complete LaTeX document containing the corrected solutions for all problems from the PDF.
                If the review indicates no mistakes, simply convert the Markdown solutions to high-quality LaTeX.
                """
                pdf_path_arg = str(original_input)
            else:
                if is_pdf_path and not solver_supports_pdf:
                    print(f"🔄 Solver {self.solver.get_name()} doesn't support PDF upload. Extracting text...")
                    final_input_data = self.extract_text_from_pdf(str(original_input))
                else:
                    final_input_data = original_input

                prompt = f"""
                You are a LaTeX expert. A previous set of solutions you generated has been reviewed, and corrections are needed.
                Apply the following fixes to the original solutions and generate a final, complete LaTeX document.
                Ensure the LaTeX is well-formatted, includes the original problem statements (commented out or clearly marked), and their corrected solutions.
                
                ### Original Problem Set
                {final_input_data}
                """
                
            prompt += f"""
            Use proper LaTeX formatting for:
            - Mathematical equations and formulas
            - Matrices (using bmatrix, pmatrix, etc.)
            - Fractions, subscripts, superscripts
            - Aligned equations where appropriate

            ### Original Solutions (in Markdown)
            {solution_text}

            ### Fix suggestions
            {fix_instructions}

            ### Formatting instructions
            {format_instructions}

            Generate the corrected solutions in a single LaTeX document. Don't omit details, preserve the correct step-by-step process in the original solutions and correct the wrong ones.
            """
            if pdf_path_arg:
                return self.solver.generate(prompt, pdf_path=pdf_path_arg)
            return self.solver.generate(prompt)
        return _generate_latex()

    def convert_markdown_to_latex(self, original_input: Union[str, List[str]], markdown_text: str, format_instructions: Optional[str] = None, is_pdf_path: bool = False) -> str:
        """Prompts an LLM to convert Markdown to LaTeX."""
        @self._get_retry_decorator()
        def _convert():
            print("📄 Converting final Markdown to LaTeX...")
            
            solver_supports_pdf = self.solver.supports_pdf_upload
            pdf_path_arg = None

            if is_pdf_path and solver_supports_pdf:
                prompt = """
                You are a LaTeX expert. Convert the following Markdown text into a well-formatted LaTeX document body.
                
                IMPORTANT: Generate ONLY the LaTeX body content for the exercises. Do NOT include a preamble, \documentclass, \begin{document}, or \end{document} tags.
                Start directly with the first \section*{Exercise...}.
                """
                pdf_path_arg = str(original_input)
            else:
                if is_pdf_path and not solver_supports_pdf:
                    print(f"🔄 Solver {self.solver.get_name()} doesn't support PDF upload. Extracting text...")
                    final_input_data = self.extract_text_from_pdf(str(original_input))
                else:
                    final_input_data = original_input

                prompt = f"""
                You are a LaTeX expert. Please convert the following Markdown text into well-formatted LaTeX body content.
                Ensure that problem statements and solutions are clearly distinguished.
                
                IMPORTANT: Generate ONLY the LaTeX body content for the exercises. Do NOT include a preamble, \documentclass, \begin{document}, or \end{document} tags.
                Start directly with the first \section*{{Exercise...}}.
                
                ### Original Problem Set
                {final_input_data}
                """
                
            prompt += f"""
            Use proper LaTeX formatting for:
            - Mathematical equations and formulas
            - Matrices (using bmatrix, pmatrix, etc.)
            - Fractions, subscripts, superscripts
            - Aligned equations where appropriate
            - don't over use bold font.

            ### Markdown Content
            {markdown_text}

            ### Formatting instructions
            {format_instructions}

            Generate a complete LaTeX document.
            """
            if pdf_path_arg:
                return self.solver.generate(prompt, pdf_path=pdf_path_arg)
            return self.solver.generate(prompt)
        return _convert()

    @staticmethod
    def save_output(filename: str, content: str):
        """Saves content to a file."""
        os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✅ Output saved to {filename}")
    
    def clear_all_caches(self):
        """Clear file caches from all providers."""
        if self.solver:
            self.solver.clear_file_cache()
        if self.verifier:
            self.verifier.clear_file_cache()

    def process(self, pdf_path: str, initial_check: bool = False) -> Optional[str]:
        """Main processing pipeline for a single PDF, returning the final LaTeX."""
        try:
            # Create a subdirectory for outputs based on PDF name
            pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
            output_dir = os.path.join("outputs", pdf_name)

            # Determine if we should use direct PDF upload or text extraction
            is_pdf_path = (self.pdf_mode == PDFMode.DIRECT_UPLOAD)
            
            # Step 1: Either use PDF directly or extract text
            if is_pdf_path:
                print("📄 Using direct PDF upload mode...")
                input_data = pdf_path
            else:
                print("📄 Using text extraction mode...")
                input_data = self.extract_text_from_pdf(pdf_path)
                self.save_output(os.path.join(output_dir, "problem_set.md"), input_data)

            # Step 2: Iterative solving
            hints = None
            if os.path.exists("hints.md"):
                print("📋 Found hints.md, incorporating it into the initial prompt.")
                with open("hints.md", "r", encoding="utf-8") as f:
                    hints = f.read()
            format_instructions = None
            if os.path.exists("format.md"):
                print("📋 Found format.md, incorporating it into the initial prompt.")
                with open("format.md", "r", encoding="utf-8") as f:
                    format_instructions = f.read()

            # Define paths for draft and review files
            draft_path = os.path.join(output_dir, "solutions_draft.md")
            review_path = os.path.join(output_dir, "review.md")

            count = 0
            for i in range(self.rounds):
                print(f"💤 Entering round {i}...")
                if i == 0:
                    # If a draft already exists, use it; otherwise, generate it
                    if os.path.exists(draft_path):
                        with open(draft_path, "r", encoding="utf-8") as f:
                            draft_md = f.read()
                        print(f"📄 Existing draft found at {draft_path}, using it.")
                    else:
                        draft_md = self.generate_initial_solutions(input_data, is_pdf_path, hints=hints)
                        self.save_output(draft_path, draft_md)
                        
                        # Track solver thoughts in memory
                        if self.memory_manager and self.summarizer:
                            solver_summary = self.summarizer.summarize_solver_output(
                                draft_md, i + 1, self.solver_provider_name, is_initial=True
                            )
                            self.memory_manager.add_solver_thoughts(pdf_path, i + 1, self.solver_provider_name, solver_summary)
                    
                    if os.path.exists(review_path) and not initial_check:
                        with open(review_path, "r", encoding="utf-8") as f:
                            review = f.read()
                        print(f"📄 Existing review found at {review_path}, using it.")
                    else:
                        review = self.verify_solutions(input_data, draft_md, is_pdf_path, hints=hints, pdf_path=pdf_path)
                        self.save_output(review_path, review)
                        
                        # Track verifier feedback in memory only for newly generated reviews
                        if self.memory_manager and self.summarizer:
                            verifier_summary = self.summarizer.summarize_verifier_output(
                                review, i + 1, self.verifier_provider_name
                            )
                            self.memory_manager.add_verifier_feedback(pdf_path, i + 1, self.verifier_provider_name, verifier_summary)
                else: 
                    draft_md = self.apply_fixes_and_regenerate(input_data, draft_md, review, is_pdf_path, pdf_path=pdf_path)
                    self.save_output(draft_path, draft_md)
                    
                    # Track refined solver thoughts in memory
                    if self.memory_manager and self.summarizer:
                        solver_summary = self.summarizer.summarize_solver_output(
                            draft_md, i + 1, self.solver_provider_name, is_initial=False
                        )
                        self.memory_manager.add_solver_thoughts(pdf_path, i + 1, self.solver_provider_name, solver_summary)
                    
                    review = self.verify_solutions(input_data, draft_md, is_pdf_path, hints=hints, pdf_path=pdf_path)
                    self.save_output(review_path, review)
                    
                    # Track verifier feedback in memory
                    if self.memory_manager and self.summarizer:
                        verifier_summary = self.summarizer.summarize_verifier_output(
                            review, i + 1, self.verifier_provider_name
                        )
                        self.memory_manager.add_verifier_feedback(pdf_path, i + 1, self.verifier_provider_name, verifier_summary)
                
                # Break early if no mistakes found in review
                if "final answer to all problems: no mistakes found" in review.lower():
                    print("✅ No mistakes found in review. Exiting iterative solving early.")
                    self.solve_complete = True
                    break
                count += 1
                print()

            print(f"✅ {count} rounds of attempts and review completed.")


            # Step 3: Check for mistakes and generate final output
            if "final answer to all problems: no mistakes found" in review.lower():
                print("👍 No mistakes found. Converting Markdown to LaTeX.")
                final_latex = self.convert_markdown_to_latex(input_data, draft_md, format_instructions, is_pdf_path)
                self.solve_complete = True
            else:
                print("⚠️ Mistakes were found. Applying fixes...")
                final_latex = self.apply_fixes_and_generate_latex(input_data, draft_md, review, format_instructions, is_pdf_path)

            final_tex_path = os.path.join(output_dir, "final_solutions.tex")
            self.save_output(final_tex_path, final_latex)

            # Step 4: Apply final styling to the LaTeX file
            if STYLING_AVAILABLE:
                try:
                    styled_output_path = os.path.join(output_dir, "final_solutions_styled.tex")
                    styler = LatexStyler(
                        input_file_path=final_tex_path,
                        output_file_path=styled_output_path
                    )
                    styler.style_file()
                except Exception as e:
                    print(f"⚠️ Could not apply custom styling: {e}")
            else:
                print("🎨 Skipping final styling as LatexStyler is not available.")
            
            if self.solve_complete:
                print("\n🎉 Workflow completed successfully!")
                
                # Track successful completion in memory
                if self.memory_manager:
                    self.memory_manager.record_pdf_result(pdf_path, True)
            else:
                print("\n⚠️ Workflow completed with unresolved issues.")
                
                # Track partial completion in memory
                if self.memory_manager:
                    self.memory_manager.record_pdf_result(pdf_path, True)  # Still successful since LaTeX was generated
            
            return final_latex

        except Exception as e:
            print(f"\n❌ An error occurred during processing of {pdf_path}: {e}")
            
            # Track failure in memory
            if self.memory_manager:
                self.memory_manager.record_pdf_result(pdf_path, False, str(e), "processing")
            
            return None
        finally:
            # Clear all file caches to free memory, but don't disconnect the shared MCP client.
            self.clear_all_caches() 