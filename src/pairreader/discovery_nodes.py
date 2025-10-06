from pairreader.schemas import PairReaderState
from pairreader.vectorestore import VectorStore
from pairreader.docparser import DocParser
from pairreader.utils import logging_verbosity, langgraph_stream_verbosity, ParamsMixin
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool, InjectedToolCallId, InjectedToolArg
from langgraph.prebuilt import InjectedState
from langgraph.types import interrupt, Command
from typing import List, Optional, Dict, Any, Annotated
import chainlit as cl
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
    
    async def _summarize(self, cluster) -> AIMessage:
        cluster_docs = '\n'.join([f"doc {i+1}:\n{doc[1]} " for i,doc in enumerate(l)])
        msgs = [
            HumanMessage(self.summarization_prompt),
            HumanMessage(cluster_docs)
        ]
        response: AIMessage = await self.llm.ainvoke(msgs)
        msgs.append(response)
        return msgs

    async def __call__(self, state: PairReaderState) -> PairReaderState:
        sampled_ids = self.vectorstore.get_sample(
            n_samples=self.n_sample, 
            p_sample=self.p_sample
        )
        clusters = await self.vectorstore.get_clusters(
            sampled_ids, 
            cluster_percentage=self.cluster_percentage, 
            min_cluster_size=self.min_cluster_size, 
            max_cluster_size=self.max_cluster_size
        )
        # parallelized summarization using llm
        tasks = [self._summarize(cluster) for cluster in clusters]
        msg_summaries = await asyncio.gather(*tasks)

        # state update
        flat_msgs = [m for msgs in msg_summaries for m in msgs]
        cluster_summaries = [summary[-1].content for summary in msg_summaries]
        state_update = {
            # "messages": flat_msgs, # explodes the context window
            "cluster_summaries": cluster_summaries
        }

        return state_update


class ReduceSummarizer:
    def __init__(self, llm_name: str, fallback_llm_name: str):
        self.llm_name = llm_name
        self.fallback_llm_name = fallback_llm_name

    @property
    def llm(self):
        return (
            init_chat_model(self.llm_name)
            .with_fallbacks([init_chat_model(self.fallback_llm_name)])
        )
    
    async def __call__(self, state: PairReaderState) -> PairReaderState:
        "summrize the map summaries"
        msgs = [
            HumanMessage("Summarize the following sub-summaries resulted following the map-reduce summarisation pattern, in a concise and informative manner."),
            HumanMessage('\n'.join((f"map-summary {i+1}:\n{summary} " for i, s in enumerate(state.cluster_summaries))))
        ]
        response: AIMessage = await self.llm.ainvoke(msgs)
        msgs.append(response)
        state_update = {
            "messages": msgs,
        }
        cl.Message(
            "Summarized the map summaries:\n"
            f"{response.content}"
        ).send()
        return state_update
