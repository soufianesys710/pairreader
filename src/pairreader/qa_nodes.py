from pairreader.schemas import PairReaderState, HITLDecision
from pairreader.vectorestore import VectorStore
from pairreader.docparser import DocParser
from pairreader.utils import Verboser, BaseNode, LLMNode, RetrievalNode
from pairreader.prompts_msgs import QA_PROMPTS, QA_MSGS
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langgraph.types import interrupt
from typing import List, Optional, Dict, Any


class QueryOptimizer(LLMNode):
    """
    Optimizes user queries for vector store retrieval.

    Expected input state keys:
        - "user_query": str, the original query from the user.

    Output state keys:
        - "subqueries": list[str], queries to use for retrieval.

    Features:
        - query_decomposition: If True, decomposes the query into sub-queries (LLM decides how many).
    """
    def __init__(self, query_decomposition: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.query_decomposition = query_decomposition

    @Verboser(verbosity_level=2)
    async def __call__(self, state: PairReaderState, *args, **kwds) -> Dict[str, List[str]]:
        """Optimize user query for retrieval."""
        if self.query_decomposition:
            msg = HumanMessage(
                QA_PROMPTS["query_decompose"].format(user_query=state['user_query'])
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
    

class HumanInTheLoopApprover(LLMNode):
    """
    Allows the user to revise LLM-generated subqueries before retrieval.

    - Prompts user to review and edit subqueries.
    - Uses original subqueries if no user input is received within timeout.
    """

    def __init__(self, **kwargs):
        """Initialize with HITLDecision structured output schema."""
        super().__init__(structured_output_schema=HITLDecision, **kwargs)

    @Verboser(verbosity_level=2)
    async def __call__(self, state: PairReaderState, *args, **kwds) -> Dict[str, Any]:
        """Request user revision of LLM subqueries."""
        user_feedback = await self.ask(
            type="text",
            message=QA_MSGS["hitl_ask_feedback"],
            timeout=90
        )
        # if the user doesn't answer at timeout
        if not user_feedback:
            await self.send(QA_MSGS["hitl_timeout"])
            state_update = {"human_in_the_loop_decision": None}
        else:
            state["messages"].append(AIMessage(content=QA_MSGS["hitl_ask_feedback"]))
            state["messages"].append(HumanMessage(user_feedback))
            decision: HITLDecision = self.llm.invoke(state["messages"])
            state_update = {"human_in_the_loop_decision": decision}
        return state_update


class InfoRetriever(RetrievalNode):
    """
    Retrieves relevant information from the vector store based on optimized queries.

    - Uses human-revised subqueries to query the vector store.
    - Returns retrieved documents and their metadata for summarization.
    """
    def __init__(self, vectorstore: VectorStore, n_documents: int = 10, **kwargs):
        super().__init__(vectorstore=vectorstore, **kwargs)
        self.n_documents = n_documents

    @Verboser(verbosity_level=2)
    async def __call__(self, state: PairReaderState, *args, **kwds) -> Dict[str, Any]:
        """Retrieve documents from vector store."""
        subqueries = [state.get("user_query")] + state.get("subqueries")
        await self.send(QA_MSGS["retriever_querying"].format(n_queries=len(subqueries)))
        results = self.vectorstore.query(query_texts=subqueries, n_documents=self.n_documents)
        await self.send(QA_MSGS["retriever_retrieved"].format(n_docs=len(results['documents'][0])))
        state_update = {
            "retrieved_documents": results["documents"][0],
            "retrieved_metadatas": results["metadatas"][0]
        }
        return state_update


class InfoSummarizer(LLMNode):
    """
    Summarizes retrieved information based on the user's original query.

    Args:
        llm_name: Name of the language model to use for summarization
    """

    def __init__(self, llm_name: str = "anthropic:claude-3-5-haiku-latest", **kwargs):
        """Initialize without fallback (original design)."""
        super().__init__(llm_name=llm_name, fallback_llm_name=None, **kwargs)

    @Verboser(verbosity_level=2)
    async def __call__(self, state: PairReaderState, *args, **kwds) -> Dict[str, Any]:
        """Summarize retrieved documents for user query."""
        await self.send(QA_MSGS["summarizer_synthesizing"].format(n_docs=len(state['retrieved_documents'])))
        retrieved_docs = "\n\n".join(state["retrieved_documents"])
        prompt = QA_PROMPTS["info_summarizer"].format(
            user_query=state['user_query'],
            retrieved_docs=retrieved_docs
        )
        state["messages"].append(HumanMessage(prompt))
        content = await self.stream(self.llm, state["messages"])
        response = AIMessage(content=content)
        state_update = {"messages": [response], "summary": response.content}
        return state_update
        