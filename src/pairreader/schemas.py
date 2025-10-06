from langgraph.graph.message import AnyMessage, add_messages
from typing_extensions import TypedDict
from typing import Annotated, List, Optional, Dict, Literal
from pydantic import BaseModel, Field

class HITLDecision(BaseModel):
    """Decision on where to route after human in the loop revision of subqueries."""
    next_node: Literal["query_optimizer", "info_retriever"] = Field(
        description="Choose 'query_optimizer' if the user wants to regenerate subqueries, or 'info_retriever' if ready to retrieve documents"
    )

class PairReaderState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    user_query: Optional[str]
    # knowledge base
    chainlit_command: Optional[str]
    # QA Agent
    subqueries: Optional[List[str]]
    human_in_the_loop_decision: Optional[HITLDecision]
    retrieved_documents: Optional[List[str]]
    retrieved_metadatas: Optional[List[Dict]]
    summary: Optional[str]
    # Discovery Agent
    cluster_summaries: Optional[List[str]]
    summary_of_summaries: Optional[str]
