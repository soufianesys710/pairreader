from pairreader.docparser import DocParser
from pairreader.vectorestore import VectorStore
from pairreader.schemas import PairReaderState
from pairreader.nodes import QueryOptimizer, HumanInTheLoopApprover, InfoRetriever, InfoSummarizer, KnowledgeBaseHandler
from langgraph.graph.state import StateGraph
from langgraph.graph import START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing import Any, Optional, Literal, Dict

class PairReaderAgent:
    def __init__(self, docparser: Optional[DocParser] = None, vectorstore: Optional[VectorStore] = None):
        self.docparser = docparser or DocParser()
        self.vectorstore = vectorstore or VectorStore()
        self.knowledge_base_handler = KnowledgeBaseHandler(self.docparser, self.vectorstore)
        self.query_optimizer = QueryOptimizer(query_decomposition=True)
        self.info_retriever = InfoRetriever(self.vectorstore)
        self.human_in_the_loop_approver = HumanInTheLoopApprover()
        self.info_summarizer = InfoSummarizer()
        self.nodes = [
            self.knowledge_base_handler,
            self.query_optimizer,
            self.info_retriever,
            self.human_in_the_loop_approver,
            self.info_summarizer
        ]
        self.checkpointer = InMemorySaver()
        self.builder = StateGraph(PairReaderState)
        self.builder.add_node("knowledge_base_handler", self.knowledge_base_handler)
        self.builder.add_node("query_optimizer", self.query_optimizer)
        self.builder.add_node("info_retriever", self.info_retriever)
        self.builder.add_node("human_in_the_loop_approver", self.human_in_the_loop_approver)
        self.builder.add_node("info_summarizer", self.info_summarizer)
        self.builder.add_edge(START, "knowledge_base_handler")
        self.builder.add_edge("knowledge_base_handler", "query_optimizer")
        self.builder.add_edge("query_optimizer", "human_in_the_loop_approver")
        self.builder.add_conditional_edges("human_in_the_loop_approver", self.route_after_human_in_the_loop_approver)
        self.builder.add_edge("info_retriever", "info_summarizer")
        self.builder.add_edge("info_summarizer", END)
        self.workflow = self.builder.compile(checkpointer=self.checkpointer)

    async def __call__(self, input: Dict, config: Dict) -> PairReaderState:
        await self.workflow.ainvoke(input=input, config=config)

    @staticmethod
    def route_after_human_in_the_loop_approver(state: PairReaderState) -> Literal["query_optimizer", "info_retriever"]:
        """Route based on structured output decision from human in the loop approver node."""
        last_message = state["messages"][-1]
        # Default to info_retriever if no decision or user didn't respond
        if hasattr(last_message, 'name') and last_message.name:
            return last_message.name
        return "info_retriever"

    def set_params(self, **params):
        for node in self.nodes:
            node.set_params(**params)
