# PairReader - Chat with Your Documents 📚

PairReader lets you upload your documents and ask questions about them. It's like having a study partner who never forgets anything!

## 🎯 How to Use

**Three usage modes:**

**📖 Chat** - Ask questions about documents you've already uploaded (just start typing!)
**✏️ Update** - Add new documents to your existing collection
**🆕 Create** - Start fresh with a new knowledge base

To **Update** or **Create**, click the buttons at startup or use commands: **/Update**, **/Create**

## 🚀 Quick Start

1. **First time?** Click **Create** to upload your documents
2. **Upload files** (supports PDF and text files, up to 5 files at a time)
3. **Ask questions** - PairReader intelligently routes your query:
   - **Most questions** (default) → QA Agent provides targeted answers
   - **Explicit exploration** (e.g., "Give me an overview" or "What are the main themes?") → Discovery Agent provides comprehensive exploration

## 💡 Tips

- **Upload related documents** to connect ideas across them
- **Ask natural questions** - the QA agent handles most requests automatically
- **For exploration, be explicit** - say "overview", "main themes", "key ideas", or "explore the documents"
- **Review subqueries** when prompted - it helps you see how your question is being analyzed
- **Use "Create"** to start fresh when switching to a completely different topic
- **Start with exploration** on new documents to understand what's available

## 🎨 Customize Your Experience

Click the **Settings** icon to adjust how PairReader's agents work. PairReader uses **3 agents**: a **Supervisor** that routes queries, a **QA Agent** (default for most questions), and a **Discovery Agent** (for overviews/exploration).

### General Settings
- **LLM model** - Choose Haiku for speed or Sonnet for power (affects all agents)

### QA Agent Settings
The QA Agent handles most of your questions:
- **Query decomposition** - Breaks complex questions into focused searches (recommended: ON)
- **Number of documents** - How many document chunks to retrieve (default: 10)

### Discovery Agent Settings
Only applies when you explicitly ask for overviews, themes, or exploration:

**Sampling (choose one approach):**
- **n_sample** - Exact number of documents to sample (set to 0 to use percentage instead)
- **p_sample** - Percentage of documents to sample (default: 0.1 = 10%)
  - ⚠️ If `n_sample` is set above 0, it takes priority over `p_sample`

**Clustering:**
- **cluster_percentage** - Controls cluster granularity (default: 0.05 = 5%)
  - Lower values = more clusters (more detailed)
  - Higher values = fewer clusters (more general)
- **min_cluster_size** - Minimum documents per cluster (0 = auto)
- **max_cluster_size** - Maximum documents per cluster (0 = auto)
  - ⚠️ Leave both at 0 for automatic sizing (recommended)

💡 **Tip:** Start with defaults and only adjust if you need more/less detail in discovery results.

## 🔬 Behind the Scenes

PairReader uses modern LLMOps practices to ensure high-quality responses. Your conversations are automatically monitored to help improve the system's accuracy and speed. This means:
- **Better answers** - We can see what works well and what doesn't
- **Faster responses** - Performance bottlenecks are identified and fixed
- **Continuous improvement** - The system gets smarter with each interaction

All monitoring is done through LangSmith, providing complete transparency into how your questions are processed.

**Happy reading!** 📖✨
