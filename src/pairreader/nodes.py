from pairreader.schemas import PairReaderState
from pairreader.vectorestore import VectorStore
from pairreader.docparser import DocParser
from pairreader.utils import logging_verbosity, langgraph_stream_verbosity
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage, AnyMessage
from langgraph.types import interrupt
from typing import List, Optional, Union, Dict, Any
import chainlit as cl


class ChainlitCommandHandler:
    """
    Handles Chainlit commands (Create, Update) and file upload logic.

    - Updates state with chainlit_command and processes file ingestion.
    - Prompts user to upload files and ingests them into the vector store.
    - Interrupts flow if no files are uploaded within timeout.
    """
    def __init__(self, docparser: DocParser, vectorstore: VectorStore):
        self.docparser = docparser
        self.vectorstore = vectorstore

    @logging_verbosity
    @langgraph_stream_verbosity
    @cl.step(type="ChainlitCommandHandler", name="ChainlitCommandHandler")
    async def __call__(self, state: PairReaderState, *args, **kwds):
        """Handle Chainlit commands and file uploads."""
        # if the user sends a command
        if (chainlit_command := state.get("chainlit_command")):
            if chainlit_command == "Create":
                self.vectorstore.flush()
            files = await cl.AskFileMessage(
                content="Please upload your files to help out reading!",
                accept=["text/plain", "application/pdf"],
                max_size_mb=10,
                max_files=5,
            ).send()
            if files is None:
                interrupt(
                    f"You haven't uploaded any files in the 60s following your {chainlit_command} command!"
                    "You can continue to use the your ciurrent knowledge base, or resend a Create or Update command described in the toolbox"
                )
            else:
                for f in files:
                    self.docparser.parse(f.path)
                    chunks = self.docparser.get_chunks()
                    metadatas = [{"fname": f.name}] * len(chunks)
                    self.vectorstore.ingest_chunks(chunks, metadatas)
                # files uploaded and parsed, ask for a user query
                interrupt(
                    f"Files uploaded: {[f.name for f in files]}, the knowledge base is ready. What do you want to know?"
                )
        # the user doesn't send a command, rather he should've sent a message, don't update the state
        else:
            return {}


class QueryOptimizer:
    """
    Optimizes user queries for vector store retrieval.

    Expected input state keys:
        - "user_query": str, the original query from the user.

    Output state keys:
        - "llm_subqueries": list[str], queries to use for retrieval.

    Features:
        - query_decomposition: If True, decomposes the query into sub-queries (LLM decides how many).
        - query_expansion: If True, expands the query for better retrieval.
        - max_expansion: int, max number of expansion queries to generate (default 10).

    Raises:
        ValueError: If query_expansion is used without query_decomposition.
    """
    def __init__(
            self, 
            llm_name: str = "anthropic:claude-3-5-haiku-latest",
            fallback_llm: str = "anthropic:claude-3-7-sonnet-latest",
            query_decomposition: bool = False,
            query_expansion: bool = False,
            max_expansion: int = 10
    ):
        self.llm_name = llm_name
        self.query_decomposition = query_decomposition
        self.query_expansion = query_expansion
        self.max_expansion = max_expansion
        self.llm = (
            init_chat_model(llm_name)
            .with_fallbacks([init_chat_model(fallback_llm)])
        )
        if self.query_expansion and not self.query_decomposition:
            raise ValueError("query_expansion can only be used if query_decomposition is True")

    @logging_verbosity
    @langgraph_stream_verbosity
    @cl.step(type="QueryOptimizer", name="QueryOptimizer")
    async def __call__(self, state: PairReaderState, *args, **kwds) -> Dict[str, List[str]]:
        """Optimize user query for retrieval."""
        subqueries = [state["user_query"]]
        if self.query_decomposition:
            decomposition_prompt = [
                SystemMessage(
                    "You are a query retrieval optimizer for vector store semantic search. "
                    "Decompose the user's query into simpler, smaller sub-queries better suited for vector store search. "
                    "Decide yourself how many sub-queries are optimal for retrieval. "
                    "Each sub-query should be on a new line for correct parsing using split('\\n'). "
                    "User Query:"
                ),
                HumanMessage(state["user_query"])
            ]
            subqueries_msg: AIMessage = self.llm.invoke(decomposition_prompt)
            subqueries += [s.strip() for s in subqueries_msg.content.split("\n") if s.strip()]
        if self.query_expansion and subqueries:
            expansion_prompt = [
                SystemMessage(
                    "You are a query retrieval optimizer for vector store semantic search. "
                    f"Use the following sub-queries to generate up to {self.max_expansion} additional semantically and synonymously similar queries for improved retrieval. "
                    "Each sub-query should be on a new line for correct parsing using split('\\n'). "
                    f"Do not generate more than {self.max_expansion} expansion queries. "
                    "Sub-queries:"
                ),
                HumanMessage("\n\n".join(subqueries))
            ]
            expansion_msg: AIMessage = self.llm.invoke(expansion_prompt)
            subqueries += [s.strip() for s in expansion_msg.content.split("\n") if s.strip()]
            subqueries = list(set(subqueries))
        return {"llm_subqueries": subqueries}
    

class ChainlitHumanReviser:
    """
    Allows the user to revise LLM-generated subqueries before retrieval.

    - Prompts user to review and edit subqueries.
    - Uses original subqueries if no user input is received within timeout.
    """
    def __init__(self):
        pass

    @logging_verbosity
    @langgraph_stream_verbosity
    @cl.step(type="ChainlitHumanReviser", name="ChainlitHumanReviser")
    async def __call__(self, state: PairReaderState, *args, **kwds) -> Dict[str, Any]:
        """Request user revision of LLM subqueries."""
        res = await cl.AskUserMessage(
            content=f"Please revise the llm genrated subqueries:\n{'\n'.join(state['llm_subqueries'])}"
        ).send()
        # if the user doesn(t answer at timeout
        if res is None:
            await cl.Message("You haven't revised the LLM genrated subqueries in the following 60s, we're using them as they are!").send()
            return {"human_subqueries": state["llm_subqueries"]}
        else:
            return {"human_subqueries": res["output"].split("\n")}


class InfoRetriever:
    """
    Retrieves relevant information from the vector store based on optimized queries.

    - Uses human-revised subqueries to query the vector store.
    - Returns retrieved documents and their metadata for summarization.
    """
    def __init__(self, vectorstore: VectorStore, n_results: int = 10):
        self.vectorstore = vectorstore
        self.n_results = n_results

    @logging_verbosity
    @langgraph_stream_verbosity
    @cl.step(type="InfoRetriever", name="InfoRetriever")
    async def __call__(self, state: PairReaderState, *args, **kwds) -> Dict[str, Any]:
        """Retrieve documents from vector store."""
        results = self.vectorstore.query(query_texts=state["human_subqueries"], n_results=self.n_results)
        state_update = {
            "retrieved_documents": results["documents"][0],
            "retrieved_metadatas": results["metadatas"][0]
        }
        return state_update


class InfoSummarizer:
    """
    Summarizes retrieved information based on the user's original query.

    Args:
        llm_name: Name of the language model to use for summarization
    """
    def __init__(self, llm_name: str = "anthropic:claude-3-5-haiku-latest"):
        self.llm_name = llm_name
        self.llm = init_chat_model(llm_name)

    @logging_verbosity
    @langgraph_stream_verbosity
    @cl.step(type="InfoSummarizer", name="InfoSummarizer")
    async def __call__(self, state: PairReaderState, *args, **kwds) -> Dict[str, Any]:
        """Summarize retrieved documents for user query."""
        msgs = [
            SystemMessage(
                "You are a helpful summarization assistant. Create a comprehensive summary "
                "of the retrieved information that directly addresses the user's query. "
                "Focus on relevant information and maintain accuracy."
            ),
            HumanMessage(f"User Query: {state['user_query']}"),
            HumanMessage("Retrieved Information:"),
            HumanMessage("\n\n".join(state["retrieved_documents"]))
        ]
        summary = self.llm.invoke(msgs)
        return {"response": summary}
        