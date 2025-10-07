from pairreader.schemas import PairReaderState
from pairreader.vectorestore import VectorStore
from pairreader.docparser import DocParser
from pairreader.utils import Verboser, ParamsMixin, UserIO
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool, InjectedToolCallId, InjectedToolArg
from langgraph.prebuilt import InjectedState
from langgraph.types import interrupt, Command
from typing import List, Optional, Dict, Any, Annotated
import asyncio

class MapSummarizer(ParamsMixin):
    def __init__(self, 
        vectorstore: VectorStore,
        n_sample: Optional[int] = None,
        p_sample: Optional[float] = 0.1,
        cluster_percentage: Optional[float] = 0.05,
        min_cluster_size: Optional[int] = None,
        max_cluster_size: Optional[int] = None,
        llm_name: str = "anthropic:claude-3-5-haiku-latest", 
        fallback_llm_name: str = "anthropic:claude-3-7-sonnet-latest"
    ):
        self.vectorstore = vectorstore
        self.n_sample = n_sample
        self.p_sample = p_sample
        self.cluster_percentage = cluster_percentage
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size
        self.llm_name = llm_name
        self.fallback_llm_name = fallback_llm_name
        self.summarization_prompt = "Summarize the following cluster of documents in a concise and informative manner."

    @property
    def llm(self):
        return (
            init_chat_model(self.llm_name)
            .with_fallbacks([init_chat_model(self.fallback_llm_name)])
        )
    
    async def _summarize(self, cluster, state) -> AIMessage:
        cluster_docs = '\n'.join([f"doc {i+1}:\n{doc[1]} " for i, doc in enumerate(cluster)])
        msg = HumanMessage(f"{self.summarization_prompt}\n\n{cluster_docs}")
        messages = list(state["messages"]) + [msg]
        response: AIMessage = await self.llm.ainvoke(messages)
        return [msg, response]

    @Verboser(verbosity_level=2)
    async def __call__(self, state: PairReaderState) -> Dict:
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
        # parallelized summarization using llm
        tasks = [self._summarize(cluster, state) for cluster in clusters]
        msg_summaries = await asyncio.gather(*tasks)

        # state update
        flat_msgs = [m for msgs in msg_summaries for m in msgs]
        cluster_summaries = [summary[-1].content for summary in msg_summaries]
        state_update = {
            # "messages": flat_msgs, # explodes the context window
            "cluster_summaries": cluster_summaries
        }

        return state_update


class ReduceSummarizer(UserIO, ParamsMixin):
    def __init__(self,
        llm_name: str = "anthropic:claude-3-5-haiku-latest",
        fallback_llm_name: str = "anthropic:claude-3-7-sonnet-latest"
    ):
        self.llm_name = llm_name
        self.fallback_llm_name = fallback_llm_name

    @property
    def llm(self):
        return (
            init_chat_model(self.llm_name)
            .with_fallbacks([init_chat_model(self.fallback_llm_name)])
        )

    @Verboser(verbosity_level=2)
    async def __call__(self, state: PairReaderState) -> Dict:
        "Summarize the map summaries"
        summaries_text = '\n'.join((f"map-summary {i+1}:\n{s} " for i, s in enumerate(state["cluster_summaries"])))
        msg = HumanMessage(f"Summarize the following sub-summaries resulted following the map-reduce summarisation pattern, in a concise and informative manner.\n\n{summaries_text}")
        messages = list(state["messages"]) + [msg]
        response: AIMessage = await self.llm.ainvoke(messages)
        state_update = {
            "messages": [msg, response],
            "summary_of_summaries": response.content
        }
        await self.send(response.content)
        return state_update
