from pairreader.schemas import PairReaderState, HITLDecision
from pairreader.vectorestore import VectorStore
from pairreader.docparser import DocParser
from pairreader.utils import Verboser, ParamsMixin, UserIO
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langgraph.types import interrupt
from typing import List, Optional, Dict, Any


class QueryOptimizer(UserIO, ParamsMixin):
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

    @Verboser(verbosity_level=2)
    # @cl.step(type="QueryOptimizer", name="QueryOptimizer")
    async def __call__(self, state: PairReaderState, *args, **kwds) -> Dict[str, List[str]]:
        """Optimize user query for retrieval."""
        if self.query_decomposition:
            # Build message with user query as HumanMessage for decomposition
            msg = HumanMessage(
                "You are a query retrieval optimizer for vector store semantic search. "
                "Decompose the following query into simpler, smaller sub-queries better suited for vector store search. "
                "Decide yourself how many sub-queries are optimal for retrieval. "
                "Each sub-query should be on a new line for correct parsing using split('\\n'). "
                f"User Query: {state['user_query']}"
            )
            messages = list(state["messages"]) + [msg]
            content = await self.stream(self.llm, messages)
            response = AIMessage(content=content)
            state_update = {
                "messages": [msg, response],
                "subqueries": [s.strip() for s in response.content.split("\n") if s.strip()]
            }
        else:
            # No decomposition, use original query
            state_update = {
                "subqueries": [state["user_query"]]
            }
        return state_update
    

class HumanInTheLoopApprover(UserIO, ParamsMixin):
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
            .with_structured_output(HITLDecision)
            .with_fallbacks([
                init_chat_model(self.fallback_llm_name)
                .with_structured_output(HITLDecision)
            ])
        )


    @Verboser(verbosity_level=2)
    # @cl.step(type="ChainlitHumanReviser", name="ChainlitHumanReviser")
    async def __call__(self, state: PairReaderState, *args, **kwds) -> Dict[str, Any]:
        """Request user revision of LLM subqueries."""
        ask_feedback_prompt = f"Please revise the LLM generated subqueries, please state explicitly if approve or disapprove these results!"
        user_feedback = await self.ask(
            type="text",
            message=ask_feedback_prompt,
            timeout=90
        )
        # if the user doesn't answer at timeout
        if not user_feedback:
            await self.send("You haven't revised the LLM generated subqueries in the following 90s, we're using them as they are!")
            state_update = {"human_in_the_loop_decision": None}
        else:
            state["messages"].append(AIMessage(content=ask_feedback_prompt))
            state["messages"].append(HumanMessage(user_feedback))
            decision: HITLDecision = self.llm.invoke(state["messages"])
            state_update = {"human_in_the_loop_decision": decision}
        return state_update


class InfoRetriever(UserIO, ParamsMixin):
    """
    Retrieves relevant information from the vector store based on optimized queries.

    - Uses human-revised subqueries to query the vector store.
    - Returns retrieved documents and their metadata for summarization.
    """
    def __init__(self, vectorstore: VectorStore, n_documents: int = 10):
        self.vectorstore = vectorstore
        self.n_documents = n_documents

    @Verboser(verbosity_level=2)
    # @cl.step(type="InfoRetriever", name="InfoRetriever")
    async def __call__(self, state: PairReaderState, *args, **kwds) -> Dict[str, Any]:
        """Retrieve documents from vector store."""
        subqueries = [state.get("user_query")] + state.get("subqueries")
        await self.send(f"Querying knowledge base with {len(subqueries)} optimized queries...")
        results = self.vectorstore.query(query_texts=subqueries, n_documents=self.n_documents)
        await self.send(f"✓ Retrieved {len(results['documents'][0])} relevant document chunks.")
        state_update = {
            "retrieved_documents": results["documents"][0],
            "retrieved_metadatas": results["metadatas"][0]
        }
        return state_update


class InfoSummarizer(UserIO, ParamsMixin):
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

    @Verboser(verbosity_level=2)
    # @cl.step(type="InfoSummarizer", name="InfoSummarizer")
    async def __call__(self, state: PairReaderState, *args, **kwds) -> Dict[str, Any]:
        """Summarize retrieved documents for user query."""
        await self.send(f"Synthesizing answer from {len(state['retrieved_documents'])} retrieved documents...")
        state["messages"].extend([
            HumanMessage(
                "You are a helpful summarization assistant. Create a comprehensive summary "
                "of the retrieved information that directly addresses the user's query. "
                "Focus on relevant information and maintain accuracy."
            ),
            HumanMessage(f"User Query: {state['user_query']}"),
            HumanMessage("Retrieved Information:"),
            HumanMessage("\n\n".join(state["retrieved_documents"]))
        ])
        content = await self.stream(self.llm, state["messages"])
        response = AIMessage(content=content)
        state_update = {"messages": [response], "summary": response.content}
        return state_update
        