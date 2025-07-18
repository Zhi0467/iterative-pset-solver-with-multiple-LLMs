import os
from llm_providers.openai_provider import OpenAIProvider
from utils.config import LLMConfig

class LatexStyler:
    """
    Refactors a LaTeX file to a specific style using an LLM with few-shot prompting.
    """

    def __init__(self, input_file_path: str, output_file_path: str = None):
        if not os.path.exists(input_file_path):
            raise FileNotFoundError(f"Input file not found: {input_file_path}")
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path or input_file_path

        config = LLMConfig()
        api_key = config.get_api_key(LLMConfig.PROVIDER_OPENAI)
        if not api_key:
            raise ValueError("OpenAI API key not found in environment.")
        
        provider_config = config.get_provider_config(LLMConfig.PROVIDER_OPENAI)
        self.provider = OpenAIProvider(api_key=api_key, **provider_config)

    def _get_styling_prompt(self, original_content: str) -> str:
        original_example = r"""
\section*{Exercise 1}

\subsection*{(a) Assuming a function \(z = f(x, y)\) is a multivariable function with \(x(r, s), y(r, s)\) functions of two variables \(r\) and \(s\). Write down the chain rule formula for \(\frac{\partial z}{\partial r}\).}

\paragraph{Solution:}
When \(x\) and \(y\) are functions of two variables, \(r\) and \(s\), we can find the partial derivatives of \(z\) with respect to \(r\) and \(s\) using a similar chain rule. To find the partial derivative of \(z\) with respect to \(r\) (\(\frac{\partial z}{\partial r}\)), we treat \(s\) as a constant and apply the chain rule.

The formula for \(\frac{\partial z}{\partial r}\) is:
\[
\frac{\partial z}{\partial r} = \frac{\partial f}{\partial x} \frac{\partial x}{\partial r} + \frac{\partial f}{\partial y} \frac{\partial y}{\partial r}
\]
Similarly, the formula for the partial derivative with respect to \(s\) would be:
\[
\frac{\partial z}{\partial s} = \frac{\partial f}{\partial x} \frac{\partial x}{\partial s} + \frac{\partial f}{\partial y} \frac{\partial y}{\partial s}
\]

\subsection*{(b) For a function \(z = f(x, y)\) with \(x(t) = 2t - t^2\) and \(y(t) = t^3 + 2\). What is \(\left.\frac{dz}{dt}\right|_{t=0}\) if \(\frac{\partial f}{\partial x}(0, 2) = -1\) and \(\frac{\partial f}{\partial y}(0, 2) = 3\)?}

\paragraph{Solution:}
We will use the chain rule formula:
\[
\frac{dz}{dt} = \frac{\partial f}{\partial x} \frac{dx}{dt} + \frac{\partial f}{\partial y} \frac{dy}{dt}
\]
We need to evaluate this expression at \(t = 0\).

\begin{enumerate}
    \item \textbf{Find the point \((x(0), y(0))\)} \\
    First, we evaluate \(x(t)\) and \(y(t)\) at \(t = 0\) to find the point \((x, y)\) at which we need the partial derivatives of \(f\).
    \begin{itemize}
        \item \(x(0) = 2(0) - 0^2 = 0\)
        \item \(y(0) = 0^3 + 2 = 2\)
    \end{itemize}
    So, at \(t = 0\), the point is \((x, y) = (0, 2)\).

    \item \textbf{Use the given partial derivatives} \\
    The problem gives us the values of the partial derivatives at this point:
    \begin{itemize}
        \item \(\frac{\partial f}{\partial x}(0, 2) = -1\)
        \item \(\frac{\partial f}{\partial y}(0, 2) = 3\)
    \end{itemize}

    \item \textbf{Find the derivatives of \(x(t)\) and \(y(t)\) with respect to \(t\)} \\
    Next, we find the derivatives of \(x(t)\) and \(y(t)\):
    \begin{itemize}
        \item \(\frac{dx}{dt} = \frac{d}{dt} (2t - t^2) = 2 - 2t\)
        \item \(\frac{dy}{dt} = \frac{d}{dt} (t^3 + 2) = 3t^2\)
    \end{itemize}

    \item \textbf{Evaluate the derivatives at \(t = 0\)} \\
    Now, we evaluate these derivatives at \(t = 0\):
    \begin{itemize}
        \item \(\left.\frac{dx}{dt}\right|_{t=0} = 2 - 2(0) = 2\)
        \item \(\left.\frac{dy}{dt}\right|_{t=0} = 3(0)^2 = 0\)
    \end{itemize}

    \item \textbf{Substitute all values into the chain rule formula} \\
    Finally, we plug all these values into the chain rule formula to find \(\frac{dz}{dt}\) at \(t=0\):
    \begin{align*}
        \left.\frac{dz}{dt}\right|_{t=0} &= \frac{\partial f}{\partial x}(0, 2) \cdot \left.\frac{dx}{dt}\right|_{t=0} + \frac{\partial f}{\partial y}(0, 2) \cdot \left.\frac{dy}{dt}\right|_{t=0} \\
        &= (-1) \cdot (2) + (3) \cdot (0) \\
        &= -2 + 0 \\
        &= -2
    \end{align*}
\end{enumerate}
\textbf{final answer:} the value of \(\left.\frac{dz}{dt}\right|_{t=0}\) is \textbf{-2}.
"""

        desired_example = r"""
\section*{Exercise 1}
\paragraph{Solution}
\subsection*{(a)}
The formula for \(\frac{\partial z}{\partial r}\) is:
\[
\frac{\partial z}{\partial r} = \frac{\partial f}{\partial x} \frac{\partial x}{\partial r} + \frac{\partial f}{\partial y} \frac{\partial y}{\partial r}
\]
Similarly, the formula for the partial derivative with respect to \(s\) would be:
\[
\frac{\partial z}{\partial s} = \frac{\partial f}{\partial x} \frac{\partial x}{\partial s} + \frac{\partial f}{\partial y} \frac{\partial y}{\partial s}
\]

\subsection*{(b)}
We will use the chain rule formula and evaluate this expression at \(t = 0\).

 First, we evaluate \(x(t)\) and \(y(t)\) at \(t = 0\) to find the point \((x, y)\) at which we need the partial derivatives of \(f\).
    \begin{itemize}
        \item \(x(0) = 2(0) - 0^2 = 0\)
        \item \(y(0) = 0^3 + 2 = 2\)
    \end{itemize}
    So, at \(t = 0\), the point is \((x, y) = (0, 2)\).

The problem gives us the values of the partial derivatives at this point:\(\frac{\partial f}{\partial x}(0, 2) = -1\), \(\frac{\partial f}{\partial y}(0, 2) = 3\).


 Next, we find the derivatives of \(x(t)\) and \(y(t)\): \(\frac{dx}{dt} = \frac{d}{dt} (2t - t^2) = 2 - 2t\), \(\frac{dy}{dt} = \frac{d}{dt} (t^3 + 2) = 3t^2\).


Now, we evaluate these derivatives at \(t = 0\):\(\left.\frac{dx}{dt}\right|_{t=0} = 2 - 2(0) = 2\), \(\left.\frac{dy}{dt}\right|_{t=0} = 3(0)^2 = 0\).


Finally, \(\frac{dz}{dt}\) at \(t=0\):
    \begin{align*}
        \left.\frac{dz}{dt}\right|_{t=0} &= \frac{\partial f}{\partial x}(0, 2) \cdot \left.\frac{dx}{dt}\right|_{t=0} + \frac{\partial f}{\partial y}(0, 2) \cdot \left.\frac{dy}{dt}\right|_{t=0} \\
        &= (-1) \cdot (2) + (3) \cdot (0) \\
        &= -2 + 0 \\
        &= -2
    \end{align*}
"""
        prompt = f"""
You are a LaTeX styling expert. Your task is to refactor the given LaTeX document to match a specific, more concise, and less hierarchical style. Please follow these guidelines:

1.  **Simplify Hierarchy**: For each problem (e.g., "Exercise 1"), use only one `\section*{{Exercise ...}}` at the beginning and one `\paragraph{{Solution}}` right after it. Do not repeat `\paragraph{{Solution}}` for sub-problems.
2.  **Reduce Itemization**: Where possible, convert verbose, step-by-step `enumerate` or `itemize` lists into more integrated paragraphs. Retain lists only if they are essential for clarity (like showing evaluation points).
3.  **Minimize Bold Font**: Avoid using `\textbf`. If emphasis is needed, use italics or other LaTeX conventions. Remove bolding from final answers.
4.  **Preserve All Math**: Do not alter any mathematical equations, formulas, or final numerical results. The logic for deriving the final answer or the proof detials should be preserved.

IMPORTANT: Generate ONLY the LaTeX body content for the exercises. Do NOT include a preamble, \documentclass, begin document, or end document tags.


Here is a guiding example of the transformation.

### Original LaTeX:
```latex
{original_example}
```

### Desired LaTeX:
```latex
{desired_example}
```

Now, please apply this exact styling transformation to the following LaTeX document. Provide only the fully refactored LaTeX code as a single block. Do not add any extra explanations or comments.

### LaTeX Document to Refactor:
```latex
{original_content}
```
"""
        return prompt

    def style_file(self):
        """
        Reads the input file, sends it to the LLM for styling, and saves the result.
        """
        print(f"🎨 Styling LaTeX file: {self.input_file_path}")
        with open(self.input_file_path, "r", encoding="utf-8") as f:
            original_content = f.read()

        prompt = self._get_styling_prompt(original_content)

        try:
            styled_content = self.provider.generate(prompt)
            if styled_content.strip().startswith("```latex"):
                styled_content = styled_content.strip()[7:]
            if styled_content.strip().endswith("```"):
                styled_content = styled_content.strip()[:-3]

        except Exception as e:
            print(f"❌ Error during styling: {e}")
            return

        with open(self.output_file_path, "w", encoding="utf-8") as f:
            f.write(styled_content)

        print(f"✅ Styled LaTeX saved to {self.output_file_path}") 