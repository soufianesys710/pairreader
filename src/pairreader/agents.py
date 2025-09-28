from pairreader.schemas import PairReaderState
from pairreader.nodes import QueryOptimizer, HumanReviser, InfoRetriever, InfoSummarizer
from pairreader.vectorestore import VectorStore
from langgraph.graph.state import StateGraph
from langgraph.graph import START, END
from typing import Any

class PairReaderAgent:
    def __init__(self, vs: VectorStore):
        self.vs = vs
        self.builder = StateGraph(PairReaderState)
        self.builder.add_node("query_optimizer", QueryOptimizer(query_decomposition=True, query_expansion=True))
        self.builder.add_node("info_retriever", InfoRetriever(vs))
        self.builder.add_node("human_reviser", HumanReviser())
        self.builder.add_node("info_summarizer", InfoSummarizer())
        self.builder.add_edge(START, "query_optimizer")
        self.builder.add_edge("query_optimizer", "human_reviser")
        self.builder.add_edge("human_reviser", "info_retriever")
        self.builder.add_edge("info_retriever", "info_summarizer")
        self.builder.add_edge("info_summarizer", END)
        self.workflow = self.builder.compile()

    def invoke(self, state: PairReaderState) -> PairReaderState:
        return self.workflow.invoke(state)
    
    def stream(self, state: PairReaderState, **kwargs) -> Any:
        return self.workflow.stream(state, **kwargs)
