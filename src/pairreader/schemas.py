from langgraph.graph.message import AnyMessage, add_messages
from typing_extensions import TypedDict
from typing import Annotated, List, Optional, Dict


class PairReaderState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    user_query: Optional[str]
    retrieval_queries: Optional[List[str]]
    retrieved_documents: Optional[List[str]]
    retrieved_metadatas: Optional[List[Dict]]
    response: Optional[str]