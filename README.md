# 🔍 Agentic Web Summarizer

A beginner-friendly application that demonstrates the concept of AI agents by summarizing web content using Large Language Models (LLMs).

## 📋 Overview

This project showcases how to build an agentic AI application that:

1. **Fetches content** from any web page
2. **Summarizes** the content using an LLM (OpenAI or Groq)
3. **Stores** the summaries in a vector database for potential future retrieval

The application is built with Streamlit for the user interface and LangChain for orchestrating the AI components.

## 🧠 What is an Agentic AI?

An "agent" in AI refers to a system that can:
- Perceive its environment (in this case, web pages)
- Make decisions based on that information (generate summaries)
- Take actions to achieve specific goals (store information for later use)

This simple application demonstrates these concepts in an approachable way.

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- API keys for the LLM providers you want to use (OpenAI and/or Groq)

### Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd agentic_web_summarizer
   ```

2. Create a virtual environment:
   ```
   python -m venv ai-env
   ```

3. Activate the virtual environment:
   - Windows: `ai-env\Scripts\activate`
   - macOS/Linux: `source ai-env/bin/activate`

4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

5. Set up your API keys in one of these ways:

   a. As environment variables:
      - For OpenAI: `export OPENAI_API_KEY=your_api_key_here`
      - For Groq: `export GROQ_API_KEY=your_api_key_here`
      - For LangChain tracing: `export LANGCHAIN_API_KEY=your_api_key_here`
   
   b. Through the UI:
      - Launch the app and enter your API keys in the sidebar under "API Keys"
      - These keys are stored in session state and not saved to disk
   
   c. Create a .env file:
      ```
      OPENAI_API_KEY=your_openai_key
      GROQ_API_KEY=your_groq_key
      LANGCHAIN_API_KEY=your_langchain_key
      ```

### Running the Application

You can start the application in two ways:

#### Option 1: Using the custom startup scripts (recommended)

These scripts handle PyTorch compatibility issues automatically:

- Windows:
  ```
  run_app.bat
  ```

- macOS/Linux:
  ```
  chmod +x run_app.sh
  ./run_app.sh
  ```

#### Option 2: Using Streamlit directly

```
streamlit run main.py
```

The application will be available at http://localhost:8501

## 🧩 Project Structure

```
agentic_web_summarizer/
├── agents/                # Agent definitions
│   ├── critique_agent.py  # Agent for evaluating summaries
│   ├── orchestrator_agent.py # Coordinates multi-agent workflow
│   ├── research_agent.py  # Agent for researching topics
│   ├── summarization_agent.py # Agent for summarizing content
│   └── web_agent.py       # Base web agent implementation
├── config/                # Configuration
│   └── database.py        # Database configuration
├── llm/                   # LLM provider integrations
│   └── llm_provider.py    # LLM provider abstraction
├── models/                # Database models
│   ├── agent_decisions.py # Model for agent decisions
│   ├── agent_log_utils.py # Utilities for logging agent actions
│   └── agent_message.py   # Model for agent communications
├── scripts/               # Utility scripts
│   ├── init_db.py         # Database initialization
│   └── view_agent_interactions.py # View agent interactions
├── tools/                 # Tools used by the agents
│   ├── fetcher.py         # Web content fetching
│   ├── summarizer.py      # Text summarization
│   └── vectorstore.py     # Vector database operations
├── data/                  # Directory for storing summaries (created at runtime)
├── main.py                # Main Streamlit application
├── run_app.py             # Custom startup script
├── run_app.bat            # Windows startup script
├── run_app.sh             # Unix/Linux/Mac startup script
└── requirements.txt       # Project dependencies
```

## 🔧 Key Components Explained

### LLM Providers

The application supports multiple LLM providers through the `llm_provider.py` module:
- **OpenAI**: Uses GPT models for high-quality summaries
- **Groq**: An alternative provider that offers fast inference

The application also supports LangChain tracing for monitoring and debugging:
- Track token usage and costs
- Debug prompts and responses
- Analyze chain execution
- Monitor performance metrics

To enable tracing, provide a LangChain API key in the UI or as an environment variable.

### Tools

The application uses several tools:

1. **Fetcher**: Retrieves and cleans content from web pages
2. **Summarizer**: Generates concise summaries using LLMs
3. **VectorStore**: Stores and retrieves summaries using vector embeddings

### Agents

The application now features a multi-agent system with specialized agents:

1. **Orchestrator Agent**: Coordinates the workflow between multiple agents
2. **Summarization Agent**: Specializes in generating summaries from web content
3. **Research Agent**: Extracts key topics and researches additional information
4. **Critique Agent**: Evaluates and improves summaries

These agents communicate with each other through a message-passing system, with all interactions logged to the database for transparency and analysis.

## 🔄 How It Works

### Single-Agent Mode

1. The user enters a URL in the Streamlit interface
2. The application fetches the content from the URL
3. The content is sent to an LLM with a prompt to generate a summary
4. The summary is displayed to the user and stored in a vector database
5. (Optional) The summary can be saved to disk for future reference

### Multi-Agent Mode

1. The user enters a URL in the Streamlit interface
2. The Orchestrator Agent delegates tasks to specialized agents:
   - Summarization Agent generates an initial summary
   - Research Agent extracts key topics and researches them
   - Critique Agent evaluates the summary quality
3. If the summary quality is below a threshold, the Critique Agent improves it
4. The final summary, research insights, and evaluation are displayed to the user
5. All agent interactions are logged to the database
6. (Optional) The results can be saved to disk for future reference

## 🛠️ Customization

You can customize the application by:
- Adding new LLM providers in `llm_provider.py`
- Creating new summary styles in `summarizer.py`
- Adding new specialized agents in the `agents/` directory
- Modifying the agent communication patterns in `orchestrator_agent.py`
- Adding new database models in the `models/` directory
- Creating new tools in the `tools/` directory

## 📚 Learning Resources

To learn more about the concepts used in this project:

- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Vector Databases Explained](https://www.pinecone.io/learn/vector-database/)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.