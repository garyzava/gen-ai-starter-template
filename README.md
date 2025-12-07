# gen-ai-starter-template

## Project Structure
```
generative_ai_project/
├── .env                <-- Secrets (API Keys) - NEVER commit this
├── .env.example        <-- Template for secrets
├── .gitignore          <-- Standard git ignore (inc. .venv, .env)
├── .python-version     <-- Pinned Python version for uv
├── pyproject.toml      <-- Definitive build & dependency config
├── uv.lock             <-- Exact dependency lockfile
├── README.md
├── Dockerfile
├── config/
│   ├── __init__.py
│   ├── settings.py     <-- Pydantic BaseSettings (validates env vars)
│   └── prompts.yaml    <-- Keep prompts separate from code
├── src/
│   ├── __init__.py
│   ├── schemas/        <-- Pydantic models (Data & I/O Validation)
│   │   ├── __init__.py
│   │   ├── chat.py     <-- Message/History schemas
│   │   └── config.py   <-- Config validation models
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base.py     <-- Abstract Base Class (ABC)
│   │   ├── client.py   <-- Async LLM Client (OpenAI/Anthropic wrapper)
│   │   └── factory.py  <-- Factory pattern to switch models easily
│   ├── vector_store/   <-- Abstraction for Embeddings/DB
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── chroma.py   <-- Example: Local ChromaDB adapter
│   ├── services/       <-- Business logic layer
│   │   ├── __init__.py
│   │   └── rag_chain.py
│   └── utils/
│       ├── __init__.py
│       ├── telemetry.py <-- Logging/Tracing (e.g., LangSmith/Otell)
│       └── decorators.py <-- Retry logic / Rate limiting
├── tests/              <-- Dedicated test suite
│   ├── __init__.py
│   ├── conftest.py     <-- Pytest fixtures
│   ├── unit/           <-- Fast tests (no API calls)
│   │   ├── test_schemas.py
│   │   └── test_utils.py
│   └── integration/    <-- Slow tests (Real API calls or mocks)
│       └── test_llm_flow.py
├── data/
│   ├── .keep           <-- Gitkeep to maintain folder structure
│   └── chroma_db/      <-- Local vector store persistence (ignored in git)
├── examples/
│   └── simple_rag.py
└── notebooks/
    └── experimentation.ipynb
```

## How to use this with uv
Since uv is your package manager, you do not need to create a virtual environment manually (no more python -m venv venv). uv handles this for you.

1. Initialize the project

Run this in your terminal at the project root:

```Bash
# This reads pyproject.toml, creates a virtual environment,
# and installs all dependencies (including dev ones)
uv sync
```

Note: This generates a uv.lock file ensuring every developer on your team uses the exact same package versions.

2. Running Tools

You don't need to activate the environment. Use uv run to execute commands inside the isolated environment:

* Run Tests:

```Bash
uv run pytestv -v
```

* Run Linter:

Checks your Python code for errors, style issues, and potential bugs
It's extremely fast (10-100x faster than traditional linters like flake8 or pylint)
Can detect issues like unused imports, undefined variables, formatting problems, etc

```Bash
uv run ruff check .
```

* Run a Script:

```Bash
# you have to create the .env and add your OpenAI API first
uv run python src/examples/verify_setup.py
```

* Start Jupyter:

```Bash
uv run jupyter notebook
```

3. Adding new packages later
If you need to add Anthropic support later, just run:

```Bash
uv add anthropic
```

This automatically updates pyproject.toml and uv.lock for you.


## Why this is good

1. The uv Workflow (pyproject.toml)
Instead of a loose requirements.txt, your pyproject.toml is now the single source of truth.

* Command: uv init creates the structure.
* Command: uv add pydantic openai installs packages and updates the file.
* Command: uv run pytest executes tests in an isolated environment automatically.

2. Pydantic Integration (src/schemas/)
We have moved away from passing Python dictionaries around. Pydantic is now a first-class citizen used for:

* Input/Output Guardrails: Ensuring the LLM returns data in the format your app expects.
* Settings Management: In config/settings.py, use pydantic-settings to load .env variables. If an API Key is missing, the app crashes immediately with a clear error, rather than failing silently later.

3. Testing Infrastructure (tests/)

* Unit Tests: Test your Pydantic schemas and utility functions. These should run in milliseconds.
* Integration Tests: Test the llm client. Tip: Use libraries like vcrpy to record HTTP responses so you don't burn API credits every time you run tests.

4. Vector Store Abstraction (src/vector_store/)

A code module that allows you to swap the backend easily. You might start with a local ChromaDB or FAISS (saved to data/), but later switch to Pinecone or Qdrant in the cloud without rewriting your main application logic.
