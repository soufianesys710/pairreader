# PairReader

A smart document companion that allows you to chat with your books, presentations, notes, and other documents. Upload your files and have conversations with your content using advanced AI capabilities.

## 🎯 What is PairReader?

PairReader is like having a study partner who never forgets anything! It lets you upload your documents and ask questions about them using natural language.

### Three Usage Modes:
- **📖 Use** - Chat with documents you've already uploaded
- **✏️ Update** - Add new documents to your collection  
- **🆕 Create** - Start fresh with new documents

All three modes are accessible via clickable buttons at startup or using commands: `/Use`, `/Update`, `/Create`

## 🚀 Quick Start

### Installation

**Prerequisites:**
- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) package manager

```bash
git clone https://github.com/soufianesys710/pairreader.git
cd pairreader
uv sync
```

### Running the Application

```bash
uv run chainlit run app.py
```

Then open your browser to `http://localhost:8000`

### How to Use

1. **Choose a mode** when you start, or use commands later
2. **Upload files** (if using Create or Update mode)
   - Supports PDF and text files
   - Up to 5 files at a time
3. **Ask questions** about your content naturally

## 💡 Tips for Best Results

- Upload related documents to connect ideas across them
- Use natural language - talk like you would to a person
- Ask specific questions for better results
- Break complex questions into simpler ones
- Use "Create" to restart when your knowledge base feels crowded

## 🔧 Technical Details

### Document Processing Pipeline
1. **Parsing**: Uses Docling's `DocumentConverter` for robust document parsing
2. **Chunking**: Implements `HybridChunker` for intelligent document segmentation  
3. **Storage**: ChromaDB vector store with persistent storage capabilities
4. **Retrieval**: Similarity-based search with metadata support

### Key Components
- **DocParser**: Handles document conversion and chunking
- **VectorStore**: Manages vector storage and query operations
- **Chainlit Interface**: Provides the web-based chat interface

## 🚧 Development

### Adding Dependencies
```bash
uv add <package-name>
```

### Development Tools
```bash
uv sync --group dev
```

## 🔮 Future Roadmap

- **Agent Architecture**: Implement multi-agent system using LangGraph
- **Enhanced Parsing**: Improved table and image extraction
- **Advanced Chunking**: Embedding and tokenization-aware chunking
- **Multi-modal Support**: Support for more file formats
- **Performance Optimization**: Enhanced retrieval and processing speed

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 🙏 Acknowledgments

- [Chainlit](https://chainlit.io/) for the chat interface framework
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Docling](https://docling-project.github.io/docling/) for document parsing
- [LangGraph](https://langchain-ai.github.io/langgraph/) for future agent architecture

---

**Happy reading!** 📖 This is an early version of PairReader. Features and documentation will evolve as the project develops.