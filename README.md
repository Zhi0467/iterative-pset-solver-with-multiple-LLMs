# Auto Problem Set Solver

This project is a Python script that uses Large Language Models (LLMs) to automatically solve problem sets from PDF files.

## Features

-   **Multiple LLM Providers:** Supports Gemini, Anthropic (Claude), OpenAI (GPT), and DeepSeek.
-   **PDF Processing:** Can either send the PDF directly to a capable model or extract the text first.
-   **Iterative Solving:** Uses a multi-round process where one LLM solves the problems and another verifies the solutions.
-   **Resilient:** Automatically retries on network errors.

## Installation

This project uses `uv` for package management.

1.  Create and activate a virtual environment:
    ```bash
    uv venv
    source .venv/bin/activate
    ```

2.  Install the project in editable mode, which will also install all dependencies:
    ```bash
    uv pip install -e .
    ```

## Usage

To run the script, use the following command:

```bash
python main.py <path_to_pdf> [pdf_mode]
```

-   `<path_to_pdf>`: The path to the PDF file containing the problem set.
-   `[pdf_mode]`: (Optional) The PDF processing mode. Can be `direct_upload` (default) or `text_extract`.

Example:
```bash
python main.py "path/to/my_problem_set.pdf"
``` 