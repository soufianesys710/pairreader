from pairreader.schemas import PairReaderState, RouteDecision
from pairreader.vectorestore import VectorStore
from pairreader.docparser import DocParser
from pairreader.utils import logging_verbosity, langgraph_stream_verbosity, ParamsMixin
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langgraph.types import interrupt
from typing import List, Optional, Dict, Any
import chainlit as cl


class KnowledgeBaseHandler(ParamsMixin):
    """
    Handles knowledge base commands (Create, Update) and file upload logic.

    - Updates state with chainlit_command and processes file ingestion.
    - Prompts user to upload files and ingests them into the vector store.
    - Interrupts flow if no files are uploaded within timeout.
    """
    def __init__(self, docparser: DocParser, vectorstore: VectorStore):
        self.docparser = docparser
        self.vectorstore = vectorstore

    @logging_verbosity
    @langgraph_stream_verbosity
    # @cl.step(type="ChainlitCommandHandler", name="ChainlitCommandHandler")
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
                await cl.Message(
                    f"You haven't uploaded any files in the 60s following your {chainlit_command} command!"
                    "You can continue to use the your ciurrent knowledge base, or resend a Create or Update command described in the toolbox"
                )
                interrupt()
            else:
                for f in files:
                    self.docparser.parse(f.path)
                    chunks = self.docparser.get_chunks()
                    metadatas = [{"fname": f.name}] * len(chunks)
                    self.vectorstore.ingest_chunks(chunks, metadatas)
                # files uploaded and parsed, ask for a user query
                await cl.Message(
                    f"Files uploaded: {[f.name for f in files]}, the knowledge base is ready. What do you want to know?"
                )
                interrupt()
        # the user doesn't send a command, rather he should've sent a message, don't update the state
        else:
            return {}


class QueryOptimizer(ParamsMixin):
    """
    Optimizes user queries for vector store retrieval.

    Expected input state keys:
        - "user_query": str, the original query from the user.

    Output state keys:
        - "subqueries": list[str], queries to use for retrieval.

    Features:
        - query_decomposition: If True, decomposes the query into sub-queries (LLM decides how many).
    """
    def __init__(
            self,
            llm_name: str = "anthropic:claude-3-5-haiku-latest",
            fallback_llm_name: str = "anthropic:claude-3-7-sonnet-latest",
            query_decomposition: bool = False
    ):
        self.llm_name = llm_name
        self.fallback_llm_name = fallback_llm_name
        self.query_decomposition = query_decomposition
    
    @property
    def llm(self):
        return (
            init_chat_model(self.llm_name)
            .with_fallbacks([init_chat_model(self.fallback_llm_name)])
        )

    @logging_verbosity
    @langgraph_stream_verbosity
    # @cl.step(type="QueryOptimizer", name="QueryOptimizer")
    async def __call__(self, state: PairReaderState, *args, **kwds) -> Dict[str, List[str]]:
        """Optimize user query for retrieval."""
        msgs = []
        if self.query_decomposition:
            # Only add SystemMessage if this is the first run (messages list is empty)
            if not state["messages"]:
                msgs.append(
                    SystemMessage(
                        "You are a query retrieval optimizer for vector store semantic search. "
                        "Decompose the user's query into simpler, smaller sub-queries better suited for vector store search. "
                        "Decide yourself how many sub-queries are optimal for retrieval. "
                        "Each sub-query should be on a new line for correct parsing using split('\\n'). "
                        "User Query:"
                    )
                )
            msgs.append(HumanMessage(state["user_query"]))
            response: AIMessage = self.llm.invoke(state["messages"] + msgs)
            msgs.append(response)
        state_update = {
            "messages": msgs,
            "subqueries": [s.strip() for s in response.content.split("\n") if s.strip()]
        }
        return state_update
    

class HumanInTheLoopApprover(ParamsMixin):
    """
    Allows the user to revise LLM-generated subqueries before retrieval.

    - Prompts user to review and edit subqueries.
    - Uses original subqueries if no user input is received within timeout.
    """
    def __init__(
        self,
        llm_name: str = "anthropic:claude-3-5-haiku-latest",
        fallback_llm_name: str = "anthropic:claude-3-7-sonnet-latest"
    ):
        self.llm_name = llm_name
        self.fallback_llm_name = fallback_llm_name

    @property
    def llm(self):
        return (
            init_chat_model(self.llm_name)
            .with_structured_output(RouteDecision)
            .with_fallbacks([
                init_chat_model(self.fallback_llm_name)
                .with_structured_output(RouteDecision)
            ])
        )


    @logging_verbosity
    @langgraph_stream_verbosity
    # @cl.step(type="ChainlitHumanReviser", name="ChainlitHumanReviser")
    async def __call__(self, state: PairReaderState, *args, **kwds) -> Dict[str, Any]:
        """Request user revision of LLM subqueries."""
        ask_feedback_prompt = f"Please revise the llm genrated subqueries!"
        user_feedback = await cl.AskUserMessage(
            content=ask_feedback_prompt + f"\n{'\n'.join(state['subqueries'])}"
        ).send()
        # if the user doesn't answer at timeout
        if user_feedback is None:
            await cl.Message("You haven't revised the LLM genrated subqueries in the following 60s, we're using them as they are!").send()
            return {}
        else:
            msgs = state["messages"] + [
                AIMessage(content=ask_feedback_prompt),
                HumanMessage(user_feedback["output"])
            ]
            decision: RouteDecision = self.llm.invoke(msgs)
            state_update = {
                "messages": [
                    AIMessage(content=ask_feedback_prompt),
                    HumanMessage(user_feedback["output"]),
                    AIMessage(content=decision.next_node, name=decision.next_node)
                ]
            }
            return state_update


class InfoRetriever(ParamsMixin):
    """
    Retrieves relevant information from the vector store based on optimized queries.

    - Uses human-revised subqueries to query the vector store.
    - Returns retrieved documents and their metadata for summarization.
    """
    def __init__(self, vectorstore: VectorStore, n_documents: int = 10):
        self.vectorstore = vectorstore
        self.n_documents = n_documents

    @logging_verbosity
    @langgraph_stream_verbosity
    # @cl.step(type="InfoRetriever", name="InfoRetriever")
    async def __call__(self, state: PairReaderState, *args, **kwds) -> Dict[str, Any]:
        """Retrieve documents from vector store."""
        subqueries = [state.get("user_query")] + state.get("subqueries")
        results = self.vectorstore.query(query_texts=subqueries, n_documents=self.n_documents)
        state_update = {
            "retrieved_documents": results["documents"][0],
            "retrieved_metadatas": results["metadatas"][0]
        }
        return state_update


class InfoSummarizer(ParamsMixin):
    """
    Summarizes retrieved information based on the user's original query.

    Args:
        llm_name: Name of the language model to use for summarization
    """
    def __init__(self, llm_name: str = "anthropic:claude-3-5-haiku-latest"):
        self.llm_name = llm_name
    
    @property
    def llm(self):
        return init_chat_model(self.llm_name)

    @logging_verbosity
    @langgraph_stream_verbosity
    # @cl.step(type="InfoSummarizer", name="InfoSummarizer")
    async def __call__(self, state: PairReaderState, *args, **kwds) -> Dict[str, Any]:
        """Summarize retrieved documents for user query."""
        msgs = [
            HumanMessage(
                "You are a helpful summarization assistant. Create a comprehensive summary "
                "of the retrieved information that directly addresses the user's query. "
                "Focus on relevant information and maintain accuracy."
            ),
            HumanMessage(f"User Query: {state['user_query']}"),
            HumanMessage("Retrieved Information:"),
            HumanMessage("\n\n".join(state["retrieved_documents"]))
        ]
        summary :AIMessage = self.llm.invoke(msgs)
        await cl.Message(content=summary.content).send()
        state_update = {"response": summary.content, "messages": msgs + [summary]}
        return state_update
        