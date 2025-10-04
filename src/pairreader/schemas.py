from langgraph.graph.message import AnyMessage, add_messages
from typing_extensions import TypedDict
from typing import Annotated, List, Optional, Dict, Literal
from pydantic import BaseModel, Field


class PairReaderState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    user_query: Optional[str]
    chainlit_command: Optional[str]
    subqueries: Optional[List[str]]
    retrieved_documents: Optional[List[str]]
    retrieved_metadatas: Optional[List[Dict]]
    response: Optional[str]


class RouteDecision(BaseModel):
    """Decision on where to route after human revision of subqueries."""
    next_node: Literal["query_optimizer", "info_retriever"] = Field(
        description="Choose 'query_optimizer' if the user wants to regenerate subqueries, or 'info_retriever' if ready to retrieve documents"
    )