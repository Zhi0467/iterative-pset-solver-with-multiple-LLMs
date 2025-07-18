import os
import concurrent.futures
from typing import List, Dict, Any, Tuple, Union
import tempfile

from processors.problem_set_solver import ProblemSetSolver, PDFMode
from processors.latex_styler import LatexStyler

class ParallelProcessor:
    """
    Processes multiple PDF batches in parallel using multiple solver-verifier pairs and aggregates the results.
    """
    def __init__(self, 
                 pdf_paths: List[str], 
                 provider_duos: Union[Tuple[str, str], List[Tuple[str, str]]],
                 pdf_mode: PDFMode = PDFMode.DIRECT_UPLOAD,
                 rounds: int = 2):
        """
        Initialize the ParallelProcessor with multiple provider duos.
        
        Args:
            pdf_paths: List of PDF file paths to process
            provider_duos: Either a single (solver, verifier) tuple or a list of (solver, verifier) tuples
            pdf_mode: PDF processing mode (direct upload or text extract)
            rounds: Number of verification rounds
        """
        self.pdf_paths = pdf_paths
        self.pdf_mode = pdf_mode
        self.rounds = rounds
        
        # Handle both single duo and multiple duos
        if isinstance(provider_duos, tuple):
            # Single duo provided - use it for all PDFs
            self.provider_duos = [provider_duos] * len(pdf_paths)
        else:
            # Multiple duos provided
            if len(provider_duos) == 1:
                # Single duo in list - repeat for all PDFs
                self.provider_duos = provider_duos * len(pdf_paths)
            elif len(provider_duos) == len(pdf_paths):
                # One duo per PDF - use as is
                self.provider_duos = provider_duos
            else:
                # Cycle through duos to match PDF count
                self.provider_duos = [provider_duos[i % len(provider_duos)] for i in range(len(pdf_paths))]
        
        # Create mapping for easy lookup
        self.pdf_to_duo = dict(zip(pdf_paths, self.provider_duos))
        
        print(f"📋 Initialized with {len(set(self.provider_duos))} unique provider duo(s):")
        for i, (solver, verifier) in enumerate(set(self.provider_duos)):
            print(f"   Duo {i+1}: Solver={solver}, Verifier={verifier}")

    def _process_one_batch(self, pdf_path: str) -> str:
        """
        Processes a single PDF batch using its assigned solver-verifier duo and styles the output.
        Returns the styled LaTeX content.
        """
        solver_provider, verifier_provider = self.pdf_to_duo[pdf_path]
        print(f"🚀 Starting processing for: {pdf_path}")
        print(f"   Using duo: Solver={solver_provider}, Verifier={verifier_provider}")
        
        try:
            # Each thread gets its own solver instance with its assigned providers
            solver = ProblemSetSolver(
                solver_provider=solver_provider,
                verifier_provider=verifier_provider,
                pdf_mode=self.pdf_mode,
                rounds=self.rounds
            )
            # The process method returns the final LaTeX content
            latex_content = solver.process(pdf_path)
            if latex_content:
                print(f"✅ Finished processing for: {pdf_path}")
                
                # Style the individual batch output
                print(f"🎨 Styling batch output for: {pdf_path}")
                try:
                    # Create temporary files for styling
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.tex', delete=False, encoding='utf-8') as temp_input:
                        temp_input.write(latex_content)
                        temp_input_path = temp_input.name
                    
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.tex', delete=False, encoding='utf-8') as temp_output:
                        temp_output_path = temp_output.name
                    
                    # Style the content
                    styler = LatexStyler(
                        input_file_path=temp_input_path,
                        output_file_path=temp_output_path
                    )
                    styler.style_file()
                    
                    # Read the styled content
                    with open(temp_output_path, 'r', encoding='utf-8') as f:
                        styled_content = f.read()
                    
                    # Clean up temporary files
                    os.unlink(temp_input_path)
                    os.unlink(temp_output_path)
                    
                    print(f"✅ Styled batch output for: {pdf_path}")
                    return styled_content
                except Exception as e:
                    print(f"⚠️ Could not style batch output for {pdf_path}: {e}")
                    print(f"📄 Using unstyled content for: {pdf_path}")
                    return latex_content
            else:
                print(f"⚠️ No content returned for: {pdf_path}")
                return ""
        except Exception as e:
            print(f"❌ An error occurred while processing {pdf_path}: {e}")
            return ""

    def run(self, output_filename: str = "final_solutions.tex"):
        """
        Runs the parallel processing workflow using multiple provider duos.
        """
        all_latex_parts = {}

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit all PDF paths to the executor
            future_to_pdf = {executor.submit(self._process_one_batch, pdf_path): pdf_path for pdf_path in self.pdf_paths}
            
            for future in concurrent.futures.as_completed(future_to_pdf):
                pdf_path = future_to_pdf[future]
                try:
                    # Get the styled result and store it
                    styled_latex = future.result()
                    all_latex_parts[pdf_path] = styled_latex
                except Exception as exc:
                    print(f"'{pdf_path}' generated an exception: {exc}")
                    all_latex_parts[pdf_path] = ""

        # Stitch the styled results together in the original order
        final_latex_content = ""
        for pdf_path in self.pdf_paths:
            final_latex_content += all_latex_parts.get(pdf_path, "") + "\n\n"

        # Define a standard LaTeX preamble
        preamble = r"""
\documentclass[11pt, a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{geometry}
\usepackage{listings}
\usepackage{xcolor}
\usepackage{hyperref}

\geometry{a4paper, margin=1in}

\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    filecolor=magenta,      
    urlcolor=cyan,
}

\definecolor{codegreen}{rgb}{0,0.6,0}
\definecolor{codegray}{rgb}{0.5,0.5,0.5}
\definecolor{codepurple}{rgb}{0.58,0,0.82}
\definecolor{backcolour}{rgb}{0.95,0.95,0.92}

\lstdefinestyle{mystyle}{
    backgroundcolor=\color{backcolour},   
    commentstyle=\color{codegreen},
    keywordstyle=\color{magenta},
    numberstyle=\tiny\color{codegray},
    stringstyle=\color{codepurple},
    basicstyle=\ttfamily\footnotesize,
    breakatwhitespace=false,         
    breaklines=true,                 
    captionpos=b,                    
    keepspaces=true,                 
    numbers=left,                    
    numbersep=5pt,                  
    showspaces=false,                
    showstringspaces=false,
    showtabs=false,                  
    tabsize=2,
    language=Python
}
\lstset{style=mystyle}

\title{Problem Set Solutions}
\author{}
\date{}

\begin{document}
\maketitle
\thispagestyle{empty}
"""
        
        # Combine preamble, styled content, and closing tag
        full_document = preamble + final_latex_content + "\n\\end{document}\n"

        # Define the output path in the 'outputs' directory
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        final_output_path = os.path.join(output_dir, output_filename)
        
        with open(final_output_path, "w", encoding="utf-8") as f:
            f.write(full_document)
        
        print(f"✅ All batches processed and styled. Final stitched LaTeX saved to: {final_output_path}")
        print(f"📝 Note: Each batch was individually styled before stitching together.")
        
        # Print summary of which duos were used
        duo_usage = {}
        for pdf_path in self.pdf_paths:
            duo = self.pdf_to_duo[pdf_path]
            if duo not in duo_usage:
                duo_usage[duo] = []
            duo_usage[duo].append(os.path.basename(pdf_path))
        
        print("\n📊 Provider duo usage summary:")
        for i, (duo, pdfs) in enumerate(duo_usage.items(), 1):
            solver, verifier = duo
            print(f"   Duo {i} (Solver={solver}, Verifier={verifier}): {len(pdfs)} PDF(s)")
            for pdf in pdfs:
                print(f"      - {pdf}") 