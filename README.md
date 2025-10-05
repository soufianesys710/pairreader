# PairReader

A smart document companion that allows you to chat with your books, presentations, notes, and other documents. Upload your files and have conversations with your content using advanced AI capabilities powered by LangGraph and Claude.

## 🎯 What is PairReader?

PairReader is like having a study partner who never forgets anything! It uses a multi-agent architecture to intelligently process your questions, optimize queries, and retrieve relevant information from your documents.

### Three Usage Modes:
- **📖 Default** - Chat with documents you've already uploaded
- **✏️ Update** - Add new documents to your existing collection
- **🆕 Create** - Start fresh with a new knowledge base

All three modes are accessible via clickable buttons at startup or using commands: `/Update`, `/Create`

## 🚀 Quick Start

### Prerequisites
- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- Anthropic API key

### Installation

```bash
git clone https://github.com/soufianesys710/pairreader.git
cd pairreader
uv sync
```

### Configuration

1. Generate a Chainlit authentication secret:
```bash
uv run chainlit create-secret
```

2. Create a `.env` file with your API keys:
```bash
ANTHROPIC_API_KEY=your_api_key_here
CHAINLIT_AUTH_SECRET=your_secret_from_step_1
```

### Running the Application

```bash
uv run pairreader
```

Then open your browser to `http://localhost:8000`

**Default credentials:** username: `admin`, password: `admin`

## 💡 How to Use

1. **Login** with the default credentials
2. **Choose a mode** when you start:
   - Click a starter button, or
   - Use `/Create` or `/Update` commands
3. **Upload files** (if using Create or Update mode)
   - Supports PDF and text files
   - Up to 5 files at a time (10MB max per file)
4. **Ask questions** about your content naturally
5. **Review and refine** - The system will show you how it's breaking down your question, and you can revise if needed

## ✨ Key Features

### Intelligent Query Processing
- **Query Decomposition**: Automatically breaks complex questions into focused sub-queries
- **Human-in-the-Loop**: Review and approve how your question is being processed
- **Multi-document Retrieval**: Searches across all your uploaded documents

### Advanced Document Processing
- **Smart Chunking**: Uses Docling's HybridChunker for intelligent document segmentation
- **Contextual Embedding**: Chunks are contextualized for better retrieval
- **Persistent Storage**: Your knowledge base is saved in ChromaDB and persists between sessions

### Configurable Settings
Adjust the following in the UI settings panel:
- **LLM Selection**: Choose between Claude Haiku (fast) or Sonnet (powerful)
- **Fallback LLM**: Automatic failover if primary model is unavailable
- **Query Decomposition**: Toggle query optimization on/off
- **Retrieval Count**: Control how many document chunks to retrieve (5-20)

## 🏗️ Architecture

PairReader uses a **LangGraph multi-agent workflow** with the following stages:

1. **Knowledge Base Handler** - Processes file uploads and manages the vector store
2. **Query Optimizer** - Decomposes complex queries into targeted sub-queries
3. **Human-in-the-Loop Approver** - Lets you review and refine the query plan
4. **Info Retriever** - Searches the vector store with optimized queries
5. **Info Summarizer** - Synthesizes retrieved information into a coherent answer

### Technology Stack
- **UI Framework**: Chainlit for interactive chat interface
- **Orchestration**: LangGraph for multi-agent workflow management
- **LLM**: Anthropic's Claude (Haiku/Sonnet) via LangChain
- **Vector Store**: ChromaDB for semantic search
- **Document Parser**: Docling for robust PDF and text processing

## 💡 Tips for Best Results

- **Start specific**: Ask focused questions rather than broad queries
- **Use context**: Reference specific topics or sections when you know them
- **Iterate**: If the first answer isn't perfect, refine your question based on what you learned
- **Organize your knowledge base**: Use Create mode to start fresh when switching to a completely different topic
- **Review subqueries**: Pay attention to how your question is decomposed—it helps you understand what's being searched

## 🔧 Development

### Adding Dependencies
```bash
uv add <package-name>
```

### Development Mode with Auto-reload
```bash
uv run chainlit run src/pairreader/__main__.py -w
```

### Development Tools
```bash
uv sync --group dev  # Includes Jupyter for experimentation
```

## 📁 Project Structure

```
pairreader/
├── src/pairreader/
│   ├── __main__.py        # Application entry point
│   ├── agents.py          # LangGraph workflow orchestration
│   ├── nodes.py           # Individual workflow nodes
│   ├── schemas.py         # State definitions
│   ├── vectorestore.py    # ChromaDB interface
│   ├── docparser.py       # Document processing
│   └── utils.py           # Decorators and utilities
├── pyproject.toml         # Project configuration
└── CLAUDE.md             # Developer documentation
```

## 🚧 Roadmap

- [ ] Enhanced table and image extraction from documents
- [ ] Embedding-aware chunking strategies
- [ ] Page number and source attribution in responses
- [ ] Secure authentication with database backend
- [ ] Support for additional document formats
- [ ] OAuth and SSO integration

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 🙏 Acknowledgments

Built with these amazing open-source projects:
- [Chainlit](https://chainlit.io/) - Interactive chat interface
- [LangGraph](https://langchain-ai.github.io/langgraph/) - Multi-agent orchestration
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [Docling](https://docling-project.github.io/docling/) - Document parsing
- [Anthropic Claude](https://www.anthropic.com/) - Language models

## 📄 License

This project is open source and available under standard terms.

---

**Happy reading with your AI pair!** 📖✨
