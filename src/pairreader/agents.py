from pairreader.docparser import DocParser
from pairreader.vectorestore import VectorStore
from pairreader.schemas import PairReaderState
from pairreader.nodes import QueryOptimizer, ChainlitHumanReviser, InfoRetriever, InfoSummarizer, ChainlitCommandHandler
from langgraph.graph.state import StateGraph
from langgraph.graph import START, END
from typing import Any, Optional

class PairReaderAgent:
    def __init__(self, docparser: Optional[DocParser] = None, vectorstore: Optional[VectorStore] = None):
        self.docparser = docparser or DocParser()
        self.vectorstore = vectorstore or VectorStore()
        self.chainlit_command_handler = ChainlitCommandHandler(self.docparser, self.vectorstore)
        self.query_optimizer = QueryOptimizer(query_decomposition=True, query_expansion=True)
        self.info_retriever = InfoRetriever(self.vectorstore)
        self.human_reviser = ChainlitHumanReviser()
        self.info_summarizer = InfoSummarizer()
        self.nodes = [
            self.chainlit_command_handler,
            self.query_optimizer,
            self.info_retriever,
            self.human_reviser,
            self.info_summarizer
        ]
        self.builder = StateGraph(PairReaderState)
        self.builder.add_node("chainlit_command_handler", self.chainlit_command_handler)
        self.builder.add_node("query_optimizer", self.query_optimizer)
        self.builder.add_node("info_retriever", self.info_retriever)
        self.builder.add_node("human_reviser", self.human_reviser)
        self.builder.add_node("info_summarizer", self.info_summarizer)
        self.builder.add_edge(START, "chainlit_command_handler")
        self.builder.add_edge("chainlit_command_handler", "query_optimizer")
        self.builder.add_edge("query_optimizer", "human_reviser")
        self.builder.add_edge("human_reviser", "info_retriever")
        self.builder.add_edge("info_retriever", "info_summarizer")
        self.builder.add_edge("info_summarizer", END)
        self.workflow = self.builder.compile()

    def __call__(self, state: PairReaderState) -> PairReaderState:
        return self.workflow.invoke(state)

    def set_params(self, **params):
        for node in self.nodes:
            node.set_params(**params)
