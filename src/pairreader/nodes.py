from pairreader.schemas import PairReaderState
from pairreader.vectorestore import VectorStore
from pairreader.docparser import DocParser
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage, AnyMessage
from langgraph.config import get_stream_writer
from langgraph.types import interrupt
from typing import List, Optional, Union, Dict, Any
import chainlit as cl
import logging

logger = logging.getLogger(__name__)


class ChainlitCommandHandler:
    """
    Handles Chainlit commands (Create, Update) and file upload logic.
    Updates state with chainlit_command and processes file ingestion.
    """
    def __init__(self, docparser: DocParser, vectorstore: VectorStore):
        self.docparser = docparser
        self.vectorstore = vectorstore

    async def __call__(self, state: PairReaderState, *args, **kwds):
        # if the user sends a command
        if (chainlit_command := state.get("chainlit_command")):
            if chainlit_command == "Create":
                logger.info("Command: Create new knowledge base")
                logger.info("Flushing knowledge base...")
                self.vectorstore.flush()
            elif chainlit_command == "Update":
                logger.info("Command: Update knowledge base")
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
                logger.info(f"Files uploaded: {[f.name for f in files]}")
                for f in files:
                    logger.info(f"Parsing file: {f.name}")
                    self.docparser.parse(f.path)
                    logger.info(f"Chunking file: {f.name}")
                    chunks = self.docparser.get_chunks()
                    logger.info(f"Ingesting chunks to the vector store, file: {f.name}")
                    metadatas = [{"fname": f.name}] * len(chunks)
                    self.vectorstore.ingest_chunks(chunks, metadatas)
                logger.info(f"Files ready: {[f.name for f in files]}")
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

    def __call__(self, state: PairReaderState, *args, **kwds) -> Dict[str, List[str]]:
        """
        Process and optimize the user query for retrieval.

        Args:
            state: Current state containing user_query

        Returns:
            Dictionary containing optimized retrieval queries
        """
        get_stream_writer()(self.__class__.__name__)
        logger.info("QueryOptimizer:")
        logger.info(f"User query: {state['user_query']}")
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
            logger.info(f"Subqueries after decomposition: {subqueries}")
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
            logger.info(f"Subqueries after expansion: {subqueries}")
        return {"llm_subqueries": subqueries}
    

class ChainlitHumanReviser:
    def __init__(self):
        pass

    async def __call__(self, state: PairReaderState, *args, **kwds) -> Dict[str, Any]:
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

    Args:
        vectorstore: VectorStore instance for document retrieval
        n_results: Maximum number of documents to retrieve (default: 10)

    Returns:
        Dictionary containing retrieved documents and their metadata
    """
    def __init__(self, vectorstore: VectorStore, n_results: int = 10):
        self.vectorstore = vectorstore
        self.n_results = n_results

    def __call__(self, state: PairReaderState, *args, **kwds) -> Dict[str, Any]:
        """
        Retrieve documents based on optimized queries.

        Args:
            state: Current state containing human_subqueries

        Returns:
            Dictionary with retrieved documents and metadata
        """
        get_stream_writer()(self.__class__.__name__)
        logger.info("InfoRetriever")
        logger.info(f"Retrieval queries: {state['human_subqueries']}")
        results = self.vectorstore.query(query_texts=state["human_subqueries"], n_results=self.n_results)
        state_update = {
            "retrieved_documents": results["documents"][0],
            "retrieved_metadatas": results["metadatas"][0]
        }
        logger.info(f"VectorStore query resulted in {len(state_update['retrieved_documents'])} documents")
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

    def __call__(self, state: PairReaderState, *args, **kwds) -> Dict[str, Any]:
        """
        Generate a summary of retrieved documents based on the user query.

        Args:
            state: Current state containing user_query and retrieved_documents

        Returns:
            Dictionary containing the generated summary
        """
        get_stream_writer()(self.__class__.__name__)
        logger.info(f"InfoSummarizer:")
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
        logger.info(f"InfoSummarizer response: {summary.content}")
        return {"response": summary}
        