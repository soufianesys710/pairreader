from pairreader.schemas import PairReaderState
from pairreader.vectorestore import VectorStore
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage, AnyMessage


class QueryOptimizer:
    def __init__(
            self, 
            llm_name: str = "anthropic:claude-3-5-haiku-latest",
            fallback_llm: str = "anthropic:claude-3-7-sonnet-latest",
            query_decomposition: bool = False, 
            query_expansion: bool = False
    ):
        self.llm_name = llm_name
        self.query_decomposition = query_decomposition
        self.query_expansion = query_expansion
        self.llm = (
            init_chat_model(llm_name)
            .with_fallbacks([init_chat_model(fallback_llm)])
        )

    def __call__(self, state: PairReaderState, *args, **kwds):
        if self.query_decomposition:
            # subqueries_msg = self.llm.invoke([
            #     SystemMessage(
            #         "Your query retrieval optimizer for vectore store semantic search purpuse, " \
            #         "you take a query from the user, you decompose into simpler, smaller, more vector store suited sub-queries" \
            #         "User Query:"
            #     ),
            #     HumanMessage(state["user_query"])
            # ])
            pass
        if self.query_expansion:
            pass
        # identity function
        return {"retrieval_queries": [state["user_query"]]}
    

class InfoRetriever:
    def __init__(self, vs: VectorStore, n_results=10):
        self.vs = vs
        self.n_results = n_results
    def __call__(self,state: PairReaderState , *args, **kwds):
        results = self.vs.query(query_texts=state["retrieval_queries"], n_results=self.n_results)
        state_update = {
            "retrieved_documents": results["documents"][0],
            "retrieved_metadatas": results["metadatas"][0]
        }
        return state_update


class InfoSummarizer:
    def __init__(self, llm_name: str = "anthropic:claude-3-5-haiku-latest"):
        self.llm_name = llm_name
        self.llm = init_chat_model(llm_name)
    def __call__(self, state: PairReaderState, *args, **kwds):
        msgs = [
            SystemMessage(
                "You're a helpful summarization assitant, summrize the information retreived from a vector store according to user query" \
                "User query:"
            ),
            HumanMessage(state["user_query"]),
            HumanMessage("Retried information"),
            HumanMessage("\n\n".join(state["retrieved_documents"]))
        ]
        summary = self.llm.invoke(msgs)
        return {"response": summary}
        