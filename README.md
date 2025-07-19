# Auto Pset Solver

An intelligent command-line tool that automatically solves problem sets (PDF files) using an orchestrated workflow with multiple LLM providers. The system analyzes PDFs, creates optimal execution plans, and processes them using dynamically selected solver-verifier pairs.

## 🚀 Features

### Intelligent Orchestration
- **Automatic PDF Analysis**: Analyzes subject, difficulty, topics, and content types
- **Dynamic Provider Selection**: Chooses optimal solver-verifier pairs based on PDF characteristics  
- **Parallel Processing**: Processes multiple PDFs concurrently for better performance
- **Adaptive Strategies**: Adjusts rounds, temperatures, and web search based on complexity

### LLM Integration
- **Multi-Provider Support**: Anthropic Claude, Google Gemini, OpenAI GPT, DeepSeek
- **PDF Processing Modes**: Direct upload or text extraction
- **Code Execution**: Built-in code execution capabilities for coding problems
- **Iterative Refinement**: Multi-round solve-verify-refine process

### Memory & Learning System
- **Session Memory**: Tracks discussions, decisions, and outcomes across sessions
- **Continuous Learning**: Generates actionable lessons for future sessions
- **Discussion Tracking**: Per-PDF discussion files with solver-verifier interactions
- **Pattern Recognition**: Identifies failure patterns and success strategies

## 📖 Usage

```bash
python main.py <path_to_pdf>
```

### Examples

**Basic usage**:
```bash
python main.py homework.pdf
```

**Multiple PDFs**:
```bash  
python main.py hw1.pdf hw2.pdf hw3.pdf
```

---

## ⚙️ Configuration

### Environment Setup
Copy `.env.example` to `.env` and configure your API keys:
```bash
ANTHROPIC_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here  
OPENAI_API_KEY=your_key_here
DEEPSEEK_API_KEY=your_key_here
```

### Recommended Settings
- **MCP Integration**: Recommended to **disable** in current version (`DEFAULT_MCP = False`)  
- **Code Execution**: Recommended to **enable** (`DEFAULT_CODE_EXECUTION = True`) - verifiers work better with code tools
- **Memory System**: Optional feature (`MEMORY_ENABLED=true/false`) - shows no significant performance effect but useful for debugging
- **Web Search**: Enable for Anthropic and Gemini providers for best results

Configure these settings in `utils/config.py` or via environment variables.

---

## ⚖️ Disclaimer

For **personal educational use**. Respect academic integrity and copyright.