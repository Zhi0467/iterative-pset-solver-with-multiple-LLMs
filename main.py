import os
import sys
import fitz  # PyMuPDF
from typing import Optional, Dict, Any, Union, List
from enum import Enum
import anthropic

from utils.config import LLMConfig
from utils.pdf_extractor import AdvancedPDFExtractor
from llm_providers.openai_provider import OpenAIProvider
from llm_providers.anthropic_provider import AnthropicProvider
from llm_providers.gemini_provider import GeminiProvider
from llm_providers.deepseek_provider import DeepSeekProvider
from llm_providers.base import BaseLLMProvider
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from httpx import ConnectError
from anthropic._exceptions import OverloadedError
from utils.pdf_extractor import AdvancedPDFExtractor
from google.genai.errors import ServerError as GeminiServerError

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
                 rounds: int = 2):
        self.config = LLMConfig()
        self.pdf_mode = pdf_mode
        self.rounds = rounds
        self.solve_complete = False
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

            solver_supports_pdf = self.solver.supports_pdf_upload
            pdf_path_arg = None
            
            if is_pdf_path and solver_supports_pdf:
                prompt = """
                You are an expert mathematician, scientist, and problem solver.
                Please solve the problems from the provided PDF with detailed, step-by-step explanations for each problem.
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
                Please solve the following problem set with detailed, step-by-step explanations for each problem.
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
            
            prompt += """
            Pay special attention to:
            - Mathematical notation and formulas
            - Matrix operations and calculations
            - Step-by-step algebraic manipulations
            - Clear explanations of each step
            - Clear presentation of answer after reasoning
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

    def verify_solutions(self, original_input: Union[str, List[str]], solution_text: str, is_pdf_path: bool = False, hints: Optional[str] = None) -> str:
        """Asks the verifier LLM to check the solutions for mistakes."""
        @self._get_retry_decorator()
        def _verify():
            print("🕵️ Asking Verifier LLM to check the solutions...")
            print(f"Using verifier: {self.verifier.get_name()}")
            
            # Check if verifier supports PDF uploads directly via the provider property
            verifier_supports_pdf = self.verifier.supports_pdf_upload
            pdf_path_arg = None
            
            if is_pdf_path and verifier_supports_pdf:
                prompt = """
                The attached PDF contains a problem set. I have drafted solutions to these problems, which are provided below.
                Please review my proposed solutions against the original problems in the PDF.
                
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
                """
                
            if hints:
                prompt += f"""
                
                ### Starter Hints
                Please incorporate these hints into your review:
                {hints}
                """

            prompt += f"""
            Identify any mistakes in the solutions, ranging from simple calculation errors to flawed reasoning.
            If you find mistakes, list them clearly and provide a concise 'fix action' for each one.
            If there are no mistakes, simply respond with "No mistakes found."

            ### Proposed Solutions
            {solution_text}

            If there are no mistakes in any of the problems, add "final answer to all problems: no mistakes found." to the end of your response.
            Output the solutions in well-formatted Markdown. 
            ONLY if there are no mistakes in any of the problems, add "final answer to all problems: no mistakes found." to the end of your response.
            OMIT THE VERIFICATION OF THE CORRECTLY SOLVED PROBLEMS. ONLY OUTPUT YOUR VERIFICATION OF THE MISTAKES.
            Please begin your review.
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

    def apply_fixes_and_regenerate(self, original_input: Union[str, List[str]], solution_text: str, fix_instructions: str, is_pdf_path: bool = False) -> str:
        """Prompts the solver LLM to apply fixes and generate a new round of solutions."""
        @self._get_retry_decorator()
        def _regenerate():
            print("⚙️ Applying fixes and re-solving...")

            solver_supports_pdf = self.solver.supports_pdf_upload
            pdf_path_arg = None
            
            if is_pdf_path and solver_supports_pdf:
                prompt = """
                You are an expert mathematician, scientist, and problem solver. The attached PDF contains a problem set.
                You previously generated solutions for this problem set, which are included below along with suggestions for corrections from a reviewer.
                Your task is to re-solve the problems from the PDF, incorporating the feedback to produce a new, corrected set of solutions.
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
                Output the solutions in well-formatted Markdown.

                ### Original Problem Set
                {final_input_data}
                """
                
            prompt += f"""
            Use proper markdown formatting, dollar sign equation environment is preferred.
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

    def apply_fixes_and_generate_latex(self, original_input: Union[str, List[str]], solution_text: str, fix_instructions: str, is_pdf_path: bool = False) -> str:
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

            Generate the corrected solutions in a single LaTeX document. Don't omit details, preserve the correct step-by-step process in the original solutions and correct the wrong ones.
            """
            if pdf_path_arg:
                return self.solver.generate(prompt, pdf_path=pdf_path_arg)
            return self.solver.generate(prompt)
        return _generate_latex()

    def convert_markdown_to_latex(self, original_input: Union[str, List[str]], markdown_text: str, is_pdf_path: bool = False) -> str:
        """Prompts an LLM to convert Markdown to LaTeX."""
        @self._get_retry_decorator()
        def _convert():
            print("📄 Converting final Markdown to LaTeX...")
            
            solver_supports_pdf = self.solver.supports_pdf_upload
            pdf_path_arg = None

            if is_pdf_path and solver_supports_pdf:
                prompt = """
                You are a LaTeX expert. Please convert the following Markdown text into a well-formatted LaTeX document.
                Ensure that problem statements from the PDF and solutions are clearly distinguished.
                """
                pdf_path_arg = str(original_input)
            else:
                if is_pdf_path and not solver_supports_pdf:
                    print(f"🔄 Solver {self.solver.get_name()} doesn't support PDF upload. Extracting text...")
                    final_input_data = self.extract_text_from_pdf(str(original_input))
                else:
                    final_input_data = original_input

                prompt = f"""
                You are a LaTeX expert. Please convert the following Markdown text into a well-formatted LaTeX document.
                Ensure that problem statements and solutions are clearly distinguished.
                
                ### Original Problem Set
                {final_input_data}
                """
                
            prompt += f"""
            Use proper LaTeX formatting for:
            - Mathematical equations and formulas
            - Matrices (using bmatrix, pmatrix, etc.)
            - Fractions, subscripts, superscripts
            - Aligned equations where appropriate

            ### Markdown Content
            {markdown_text}

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

    def process(self, pdf_path: str, initial_check: bool = False):
        """Main sub_processing pipeline."""
        try:
            # Determine if we should use direct PDF upload or text extraction
            is_pdf_path = (self.pdf_mode == PDFMode.DIRECT_UPLOAD)
            
            # Step 1: Either use PDF directly or extract text
            if is_pdf_path:
                print("📄 Using direct PDF upload mode...")
                input_data = pdf_path
            else:
                print("📄 Using text extraction mode...")
                input_data = self.extract_text_from_pdf(pdf_path)
                self.save_output("outputs/problem_set.md", input_data)

            # Step 2: Iterative solving
            hints = None
            if os.path.exists("hints.md"):
                print("📋 Found hints.md, incorporating it into the initial prompt.")
                with open("hints.md", "r", encoding="utf-8") as f:
                    hints = f.read()
            count = 0
            for i in range(self.rounds):
                print(f"💤 Entering round {i}...")
                if i == 0:
                    # If a draft already exists, use it; otherwise, generate it
                    draft_path = "outputs/solutions_draft.md"
                    if os.path.exists(draft_path):
                        with open(draft_path, "r", encoding="utf-8") as f:
                            draft_md = f.read()
                        print(f"📄 Existing draft found at {draft_path}, using it.")
                    else:
                        draft_md = self.generate_initial_solutions(input_data, is_pdf_path, hints=hints)
                        self.save_output(draft_path, draft_md)
                    review_path = "outputs/review.md"
                    if os.path.exists(review_path) and not initial_check:
                        with open(review_path, "r", encoding="utf-8") as f:
                            review = f.read()
                        print(f"📄 Existing review found at {review_path}, using it.")
                    else:
                        review = self.verify_solutions(input_data, draft_md, is_pdf_path, hints=hints)
                        self.save_output(review_path, review)
                else: 
                    draft_md = self.apply_fixes_and_regenerate(input_data, draft_md, review, is_pdf_path)
                    self.save_output("outputs/solutions_draft.md", draft_md)
                    review = self.verify_solutions(input_data, draft_md, is_pdf_path, hints=hints)
                    self.save_output("outputs/review.md", review)
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
                final_latex = self.convert_markdown_to_latex(input_data, draft_md, is_pdf_path)
                self.solve_complete = True
            else:
                print("⚠️ Mistakes were found. Applying fixes...")
                final_latex = self.apply_fixes_and_generate_latex(input_data, draft_md, review, is_pdf_path)

            self.save_output("outputs/final_solutions.tex", final_latex)
            if self.solve_complete:
                print("\n🎉 Workflow completed successfully!")

        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            raise
        finally:
            # Clear all file caches to free memory
            self.clear_all_caches()
        


def main():
    """Entry point of the script."""

    if len(sys.argv) < 2 or len(sys.argv) > 4:
        print("Usage: python main.py <path_to_pdf> [pdf_mode] [web_search_mode]")
        print("pdf_mode options: direct_upload (default), text_extract")
        print("web_search_mode options: default, all_enabled, all_disabled, solver_only, verifier_only")
        sys.exit(1)

    pdf_file_path = sys.argv[1]
    
    # Parse PDF mode from command line or use default
    pdf_mode = PDFMode.DIRECT_UPLOAD
    if len(sys.argv) >= 3:
        try:
            pdf_mode = PDFMode(sys.argv[2])
        except ValueError:
            print("Invalid PDF mode. Using default: direct_upload")
    
    # Parse web search mode from command line or use default
    web_search_mode = "default"
    if len(sys.argv) >= 4:
        web_search_mode = sys.argv[3].lower()
        if web_search_mode not in ["default", "all_enabled", "all_disabled", "solver_only", "verifier_only"]:
            print("Invalid web search mode. Using default.")
            web_search_mode = "default"
    
    # Configure web search settings for different provider combinations
    config = LLMConfig()
    

    if web_search_mode == "all_enabled":
        # Enable web search for all providers that support it
        config.set_web_search_enabled(LLMConfig.PROVIDER_ANTHROPIC, True)
        config.set_web_search_enabled(LLMConfig.PROVIDER_GEMINI, True)
        config.set_web_search_enabled(LLMConfig.PROVIDER_OPENAI, True)    # Not supported yet
        config.set_web_search_enabled(LLMConfig.PROVIDER_DEEPSEEK, True)  # Not supported yet
        
    elif web_search_mode == "all_disabled":
        # Disable web search for all providers
        config.set_web_search_enabled(LLMConfig.PROVIDER_ANTHROPIC, False)
        config.set_web_search_enabled(LLMConfig.PROVIDER_GEMINI, False)
        config.set_web_search_enabled(LLMConfig.PROVIDER_OPENAI, False)
        config.set_web_search_enabled(LLMConfig.PROVIDER_DEEPSEEK, False)
        
    elif web_search_mode == "solver_only":
        # Enable web search only for solver providers (Anthropic in Duo 1, Gemini in Duo 2)
        config.set_web_search_enabled(LLMConfig.PROVIDER_ANTHROPIC, True)   # Solver in Duo 1
        config.set_web_search_enabled(LLMConfig.PROVIDER_GEMINI, True)      # Solver in Duo 2
        config.set_web_search_enabled(LLMConfig.PROVIDER_OPENAI, False)     # Not supported yet
        config.set_web_search_enabled(LLMConfig.PROVIDER_DEEPSEEK, False)   # Not supported yet
        
    elif web_search_mode == "verifier_only":
        # Enable web search only for verifier providers (Gemini in Duo 1, OpenAI in Duo 2)
        config.set_web_search_enabled(LLMConfig.PROVIDER_ANTHROPIC, False)  # Solver in Duo 1
        config.set_web_search_enabled(LLMConfig.PROVIDER_GEMINI, True)      # Verifier in Duo 1
        config.set_web_search_enabled(LLMConfig.PROVIDER_OPENAI, False)     # Not supported yet
        config.set_web_search_enabled(LLMConfig.PROVIDER_DEEPSEEK, False)   # Not supported yet
        
    else:  # default mode
        # Balanced approach: Enable web search for verification but not for solving
        # This avoids external information influencing the core problem solving
        # but allows verification to look up mathematical concepts if needed
        config.set_web_search_enabled(LLMConfig.PROVIDER_ANTHROPIC, True)  # Solver - focused
        config.set_web_search_enabled(LLMConfig.PROVIDER_GEMINI, True)      # Verifier - can research
        config.set_web_search_enabled(LLMConfig.PROVIDER_OPENAI, False)     # Not supported yet
        config.set_web_search_enabled(LLMConfig.PROVIDER_DEEPSEEK, False)   # Not supported yet
    
    # Display current web search configuration
    print("\n📊 Current web search configuration:")
    web_search_status = config.get_web_search_status()
    for provider, enabled in web_search_status.items():
        print(f"  {provider}: {'✅ ENABLED' if enabled else '❌ DISABLED'}")
    print()
    
    # Define provider combinations
    solver1 = LLMConfig.PROVIDER_OPENAI
    solver2 = LLMConfig.PROVIDER_DEEPSEEK
    verifier1 = LLMConfig.PROVIDER_ANTHROPIC
    verifier2 = LLMConfig.PROVIDER_DEEPSEEK

    print(f"🤖 Duo 1: {solver1} (solver) + {verifier1} (verifier)")
    print(f"🤖 Duo 2: {solver2} (solver) + {verifier2} (verifier)")
    print()

    duo1 = ProblemSetSolver(
        solver_provider=solver1,
        verifier_provider=verifier1,
        pdf_mode=pdf_mode,
        rounds=1
    )

    duo2 = ProblemSetSolver(
        solver_provider=solver2,
        verifier_provider=verifier2,
        pdf_mode=pdf_mode,
        rounds=1
    )
    
    duo1.process(pdf_file_path)
    if duo1.solve_complete:
        print("🎉 Duo 1 completed successfully!")
        print()
    else:
        print("❌ Duo 1 failed to solve the problem set. Calling Duo 2...")
        print()
        duo2.process(pdf_file_path, initial_check=True)
        if duo2.solve_complete:
            print("🎉 Duo 2 completed successfully!")
        else:
            print("❌ Seems like there is at least one mistake in the problem set. Please either check the solutions manually or try again (e.g. change the solver or verifier, or run the script again).")



if __name__ == "__main__":
    main()
