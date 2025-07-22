# Auto Pset Solver


An intelligent command-line tool that automatically solves problem sets (PDF files) using an orchestrated workflow of multiple LLM providers. The system analyzes each PDF to determine the optimal solver-verifier pair and execution strategy.

## 🚀 Features

### 🧠 Intelligent Orchestration
- **Smart PDF Analysis**: Uses GPT-4 to analyze problem sets for subject, difficulty, topics, and requirements
- **Dynamic Provider Selection**: Automatically selects optimal solver-verifier pairs based on content analysis
- **Adaptive Execution Plans**: Configures rounds, temperature, and tools based on problem complexity

### 🔧 Advanced Tool Integration
- **Web Search**: Real-time web search for current information and verification
- **MCP Math Tools**: Advanced mathematical computation via Model Context Protocol server
- **Code Execution**: Secure sandbox environment for running and testing code
- **Bash Tools**: Command-line tool execution for system operations


### 📊 Multi-Provider Support
- **Anthropic Claude**: Advanced reasoning with tool use capabilities
- **OpenAI GPT**: Strategic planning and analysis
- **Google Gemini**: Visual processing and multimodal understanding
- **DeepSeek**: Cost-effective solving and verification

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
- **Prompt Caching**: Anthropic provider includes intelligent caching for 90% cost savings on repeated content

### Memory & Learning System
- **Session Memory**: Tracks discussions, decisions, and outcomes across sessions
- **Continuous Learning**: Generates actionable lessons for future sessions
- **Discussion Tracking**: Per-PDF discussion files with solver-verifier interactions
- **Pattern Recognition**: Identifies failure patterns and success strategies

## 📖 Usage

The orchestrator automatically analyzes your PDFs and creates optimal execution plans:

```bash
python main.py <pdf_path1> [pdf_path2] [pdf_path3] ...
```

### Examples

**Single PDF processing**:
```bash
python main.py homework.pdf
```

**Multiple PDF batch processing**:
```bash
python main.py hw1.pdf hw2.pdf hw3.pdf
```

### How It Works

The system operates through a sophisticated three-phase workflow:

#### 1. Analysis Phase
The [`orchestrator`](processors/orchestrator.py) uses GPT-4-Turbo as an analyst to deeply examine each PDF:

- **Subject Classification**: Identifies the primary academic domain (Mathematics, Physics, Computer Science, Humanities, Theory, or Mixed)
- **Topic Extraction**: Lists specific topics covered (e.g., "Linear Algebra", "Quantum Mechanics", "Data Structures")
- **Difficulty Assessment**: Categorizes complexity from High School through Research level
- **Content Characteristics**: Detects presence of figures, code snippets, mathematical formulas
- **Tool Requirements**: Determines if problems need current information, computational tools, or code execution
- **Scope Analysis**: Estimates word count and problem complexity

#### 2. Strategic Planning Phase
GPT-4o acts as a strategist to create optimal execution plans based on the analysis:

- **Provider Selection Logic**:
  - Problems without figures → Gemini + DeepSeek (cost-effective for text-only)
  - Simple tasks → OpenAI + OpenAI (consistent reasoning)
  - Complex problems with figures → Anthropic + Gemini (advanced multimodal capabilities)
  
- **Dynamic Configuration**:
  - **Rounds**: High School (1), Undergraduate (2), Graduate/Research (3)
  - **Temperature Tuning**: Lower for math/physics (0.0-0.2), moderate for creative tasks (0.3-0.5)
  - **Tool Enablement**: Web search for current info, MCP math tools for calculations, code execution for programming problems

#### 3. Execution Phase
The [`parallel processor`](processors/parallel_processor.py) executes the plans with advanced capabilities:

- **Concurrent Processing**: Multiple PDFs processed simultaneously with different provider configurations
- **Tool Integration**: Each provider can access web search, mathematical computation servers, and secure code execution environments
- **Intelligent Fallbacks**: If a provider fails, the system automatically retries with alternative configurations


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
- **Prompt Caching**: Automatically enabled for Anthropic - caches tools, system messages, PDFs, and large text for cost optimization

Configure these settings in `utils/config.py` or via environment variables.

### Environment Variables

The system supports various API endpoints and configurations via environment variables. See [`utils/config.py`](utils/config.py) for the complete list.

---

## 🛠️ Architecture

### Core Components

- **[`main.py`](main.py)**: Entry point with environment validation
- **[`processors/orchestrator.py`](processors/orchestrator.py)**: Intelligent workflow orchestration
- **[`processors/parallel_processor.py`](processors/parallel_processor.py)**: Multi-PDF batch processing
- **[`processors/problem_set_solver.py`](processors/problem_set_solver.py)**: Core solving logic
- **[`llm_providers/`](llm_providers/)**: Provider implementations with tool integration

### Tool Integration

- **[`mcp_server/math_tools.py`](mcp_server/math_tools.py)**: Advanced mathematical computation server
- **[`utils/sandbox.py`](utils/sandbox.py)**: Secure code execution environment
- **[`utils/network_checker.py`](utils/network_checker.py)**: Connectivity and web search validation

---

## 🔧 Troubleshooting

### Common Issues

1. **Missing API Keys**: Ensure all required API keys are set in environment variables
2. **Tool Failures**: Check network connectivity for web search and MCP server status
3. **PDF Processing**: Verify PDF files are readable and not password-protected
4. **Memory Issues**: Large PDFs may require increased system memory

---

## ⚖️ Disclaimer

For **personal educational use only**. Please respect academic integrity policies and copyright laws when using this tool.

