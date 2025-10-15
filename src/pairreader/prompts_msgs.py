"""
Prompts and Messages for PairReader agents.

This module contains:
- PROMPTS: Templates sent to LLMs for processing
- MSGS: User-facing messages for communication

This separation ensures clear distinction between AI processing and user interaction.
"""

# ============================================================================
# LLM PROMPTS - Sent to language models for processing
# ============================================================================

DISCOVERY_PROMPTS = {
    "map_summarize_cluster": "Summarize the following cluster of documents in a concise and informative manner.\n\n{cluster_docs}",
    "reduce_summaries": "Summarize the following sub-summaries resulted following the map-reduce summarisation pattern, in a concise and informative manner.\n\n{summaries_text}",
}

PAIRREADER_PROMPTS = {
    "qa_discovery_router": """You are a pair-reader agent that helps users chat with information from a knowledge base containing their uploaded documents.
You have two sub-agents: QAAgent (DEFAULT) and DiscoveryAgent (SPECIAL CASES ONLY).

**QAAgent (DEFAULT)** - Use for ALL regular questions and information requests:
- Any question seeking specific information from the documents
- Questions asking "what", "how", "why", "when", "where" about content
- Requests to explain concepts, summarize specific topics, or find information
- Examples: "What does this say about X?", "Explain Y", "How many Z are mentioned?"

**DiscoveryAgent (SPECIAL CASES ONLY)** - Use ONLY when user explicitly requests exploration:
- User explicitly asks for: "overview", "explore", "discover", "main themes", "main ideas", "key ideas", "overall summary"
- User wants high-level exploration without specific questions
- Examples: "Give me an overview", "What are the main themes?", "Explore the documents"

IMPORTANT: Default to QAAgent unless the user explicitly uses exploration keywords.
Most queries should go to QAAgent - it handles all regular information requests.

User query: {user_query}"""
}

QA_PROMPTS = {
    "query_decompose": """You are a query retrieval optimizer for vector store semantic search. Decompose the following query into simpler, smaller sub-queries better suited for vector store search. Decide yourself how many sub-queries are optimal for retrieval. Each sub-query should be on a new line for correct parsing using split('\\n'). User Query: {user_query}""",
    "info_summarizer": """You are a helpful summarization assistant. Create a comprehensive summary of the retrieved information that directly addresses the user's query. Focus on relevant information and maintain accuracy.

User Query: {user_query}

Retrieved Information:
{retrieved_docs}""",
}

# ============================================================================
# USER MESSAGES - Sent to users for communication
# ============================================================================

DISCOVERY_MSGS = {
    "map_retrieving": "Retrieving and clustering document content...",
    "map_generating": "Generating summaries for {n_clusters} clusters...",
    "reduce_synthesizing": "Synthesizing final overview from cluster summaries...",
}

PAIRREADER_MSGS = {
    "kb_flushing": "Flushing knowledge base...",
    "kb_upload_files": "Please upload your files to help out reading!",
    "kb_timeout": "You haven't uploaded any files in the 60s following your {user_command} command! You can continue to use the your current knowledge base, or resend a Create or Update command described in the toolbox",
    "kb_success": "✓ Files uploaded: {file_names}. Knowledge base now contains {len_docs} document chunks. What do you want to know?",
    "kb_processing": "Processing {n_files} file(s)...",
    "kb_parsing": "Parsing {file_name}...",
    "kb_ingesting": "Ingesting {n_chunks} chunks from {file_name}...",
}

QA_MSGS = {
    "hitl_ask_feedback": "Please revise the LLM generated subqueries, please state explicitly if approve or disapprove these results!",
    "hitl_timeout": "You haven't revised the LLM generated subqueries in the following 90s, we're using them as they are!",
    "retriever_querying": "Querying knowledge base with {n_queries} optimized queries...",
    "retriever_retrieved": "✓ Retrieved {n_docs} relevant document chunks.",
    "summarizer_synthesizing": "Synthesizing answer from {n_docs} retrieved documents...",
}
