from pairreader.schemas import PairReaderState
from pairreader.vectorestore import VectorStore
from pairreader.docparser import DocParser
from pairreader.utils import Verboser, LLMNode, RetrievalNode
from pairreader.prompts_msgs import DISCOVERY_PROMPTS, DISCOVERY_MSGS
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool, InjectedToolCallId, InjectedToolArg
from langgraph.prebuilt import InjectedState
from langgraph.types import interrupt, Command
from typing import List, Optional, Dict, Any, Annotated
import asyncio

class ClusterRetriever(RetrievalNode):
    """
    Retrieves and clusters documents from vector store.

    - Samples documents from vectorstore
    - Clusters them using semantic similarity
    """
    def __init__(
        self,
        vectorstore: VectorStore,
        n_sample: Optional[int] = None,
        p_sample: Optional[float] = 0.1,
        cluster_percentage: Optional[float] = 0.05,
        min_cluster_size: Optional[int] = None,
        max_cluster_size: Optional[int] = None,
        **kwargs
    ):
        super().__init__(vectorstore=vectorstore, **kwargs)
        self.n_sample = n_sample
        self.p_sample = p_sample
        self.cluster_percentage = cluster_percentage
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size

    @Verboser(verbosity_level=2)
    async def __call__(self, state: PairReaderState) -> Dict:
        """Samples and clusters documents from vectorstore."""
        await self.send(DISCOVERY_MSGS["map_retrieving"])
        sampled_ids = self.vectorstore.get_sample(
            n_samples=self.n_sample,
            p_samples=self.p_sample
        )
        clusters = await self.vectorstore.get_clusters(
            sampled_ids,
            cluster_percentage=self.cluster_percentage,
            min_cluster_size=self.min_cluster_size,
            max_cluster_size=self.max_cluster_size
        )
        state_update = {"clusters": clusters}
        return state_update


class MapSummarizer(LLMNode):
    """
    Summarizes document clusters in parallel using LLM.

    - Expects clusters from state
    - Generates summaries for each cluster in parallel
    """

    async def summarize_cluster(self, cluster, state) -> List:
        """Summarizes a single cluster of documents using LLM."""
        cluster_docs = '\n'.join([f"doc {i+1}:\n{doc[1]} " for i, doc in enumerate(cluster)])
        prompt = DISCOVERY_PROMPTS["map_summarize_cluster"].format(cluster_docs=cluster_docs)
        msg = HumanMessage(prompt)
        messages = list(state["messages"]) + [msg]
        response: AIMessage = await self.llm.ainvoke(messages)
        return [msg, response]

    @Verboser(verbosity_level=2)
    async def __call__(self, state: PairReaderState) -> Dict:
        """Summarize all clusters in parallel."""
        clusters = state.get("clusters", [])
        await self.send(DISCOVERY_MSGS["map_generating"].format(n_clusters=len(clusters)))

        tasks = [self.summarize_cluster(cluster, state) for cluster in clusters]
        msg_summaries = await asyncio.gather(*tasks)

        # state update
        cluster_summaries = [summary[-1].content for summary in msg_summaries]
        state_update = {
            "cluster_summaries": cluster_summaries
        }
        return state_update


class ReduceSummarizer(LLMNode):
    """
    Reduces cluster summaries into final overview.

    - Combines summaries from MapSummarizer
    - Generates comprehensive knowledge base overview
    """

    @Verboser(verbosity_level=2)
    async def __call__(self, state: PairReaderState) -> Dict:
        """Reduces cluster summaries into a final overview using LLM."""
        await self.send(DISCOVERY_MSGS["reduce_synthesizing"])
        summaries_text = '\n'.join((f"map-summary {i+1}:\n{s} " for i, s in enumerate(state["cluster_summaries"])))
        prompt = DISCOVERY_PROMPTS["reduce_summaries"].format(summaries_text=summaries_text)
        msg = HumanMessage(prompt)
        messages = list(state["messages"]) + [msg]
        content = await self.stream(self.llm, messages)
        response = AIMessage(content=content)
        state_update = {
            "messages": [msg, response],
            "summary_of_summaries": response.content
        }
        return state_update
