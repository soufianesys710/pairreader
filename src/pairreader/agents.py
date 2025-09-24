from pairreader.schemas import PairReaderState
from pairreader.nodes import QueryOptimizer, InfoRetriever, InfoSummarizer
from pairreader.vectorestore import VectorStore
from langgraph.graph.state import StateGraph
from langgraph.graph import START, END

class PairReaderAgent:
    def __init__(self, vs: VectorStore):
        self.vs = vs
        self.graph_builder = StateGraph(PairReaderState)
        self.graph_builder.add_node("query_optimizer", QueryOptimizer(query_decomposition=True, query_expansion=True))
        self.graph_builder.add_node("info_retriever", InfoRetriever(vs))
        self.graph_builder.add_node("info_summarizer", InfoSummarizer())
        self.graph_builder.add_edge(START, "query_optimizer")
        self.graph_builder.add_edge("query_optimizer", "info_retriever")
        self.graph_builder.add_edge("info_retriever", "info_summarizer")
        self.graph_builder.add_edge("info_summarizer", END)
        self.graph = self.graph_builder.compile()

    def __call__(self, state: PairReaderState) -> PairReaderState:
        return self.graph.invoke(state)
