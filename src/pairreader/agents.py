from typing import Literal

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START

from pairreader.discovery_nodes import ClusterRetriever, MapSummarizer, ReduceSummarizer
from pairreader.docparser import DocParser
from pairreader.pairreader_nodes import KnowledgeBaseHandler, QADiscoveryRouter
from pairreader.qa_nodes import (
    HumanInTheLoopApprover,
    InfoRetriever,
    InfoSummarizer,
    QueryOptimizer,
)
from pairreader.schemas import PairReaderState
from pairreader.utils import BaseAgent
from pairreader.vectorestore import VectorStore


class PairReaderAgent(BaseAgent):
    def __init__(self, docparser: DocParser | None = None, vectorstore: VectorStore | None = None):
        self.docparser = docparser or DocParser()
        self.vectorstore = vectorstore or VectorStore()
        super().__init__(
            PairReaderState,
            [
                ("knowledge_base_handler", KnowledgeBaseHandler(self.docparser, self.vectorstore)),
                ("qa_discovery_router", QADiscoveryRouter()),
                ("qa_agent", QAAgent(self.vectorstore)),
                ("discovery_agent", DiscoveryAgent(self.vectorstore)),
            ],
        )
        self.checkpointer = InMemorySaver()
        self.builder.add_edge(START, "knowledge_base_handler")
        self.builder.add_edge("knowledge_base_handler", "qa_discovery_router")
        self.builder.add_edge("qa_agent", END)
        self.builder.add_edge("discovery_agent", END)
        self.workflow = self.builder.compile(checkpointer=self.checkpointer)


# TODO: the algorithm to sample-cluster the data in knowledge base doesn't ensure entire data is covered.
class DiscoveryAgent(BaseAgent):
    def __init__(self, vectorstore: VectorStore):
        self.vectorstore = vectorstore
        super().__init__(
            PairReaderState,
            [
                ("cluster_retriever", ClusterRetriever(vectorstore=self.vectorstore)),
                ("map_summarizer", MapSummarizer()),
                ("reduce_summarizer", ReduceSummarizer()),
            ],
        )
        self.checkpointer = InMemorySaver()
        self.builder.add_edge(START, "cluster_retriever")
        self.builder.add_edge("cluster_retriever", "map_summarizer")
        self.builder.add_edge("map_summarizer", "reduce_summarizer")
        self.builder.add_edge("reduce_summarizer", END)
        self.workflow = self.builder.compile(checkpointer=self.checkpointer)


class QAAgent(BaseAgent):
    def __init__(self, vectorstore: VectorStore):
        self.vectorstore = vectorstore
        super().__init__(
            PairReaderState,
            [
                ("query_optimizer", QueryOptimizer(query_decomposition=True)),
                ("info_retriever", InfoRetriever(self.vectorstore)),
                ("human_in_the_loop_approver", HumanInTheLoopApprover()),
                ("info_summarizer", InfoSummarizer()),
            ],
        )
        self.checkpointer = InMemorySaver()
        self.builder.add_edge(START, "query_optimizer")
        self.builder.add_edge("query_optimizer", "human_in_the_loop_approver")
        self.builder.add_conditional_edges(
            "human_in_the_loop_approver", self.route_after_human_in_the_loop_approver
        )
        self.builder.add_edge("info_retriever", "info_summarizer")
        self.builder.add_edge("info_summarizer", END)
        self.workflow = self.builder.compile(checkpointer=self.checkpointer)

    @staticmethod
    def route_after_human_in_the_loop_approver(
        state: PairReaderState,
    ) -> Literal["query_optimizer", "info_retriever"]:
        """Route based on structured output decision from human in the loop approver node."""
        if (
            state.get("human_in_the_loop_decision")
            and state["human_in_the_loop_decision"].next_node
        ):
            return state["human_in_the_loop_decision"].next_node
        else:
            return "info_retriever"
