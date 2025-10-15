from typing import Annotated, Literal

from langgraph.graph.message import AnyMessage, add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class HITLDecision(BaseModel):
    """Decision on where to route after human in the loop revision of subqueries."""
    next_node: Literal["query_optimizer", "info_retriever"] = Field(
        description="Choose 'query_optimizer' if the user wants to regenerate subqueries, or 'info_retriever' if ready to retrieve documents"
    )

class PairReaderState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_query: str | None
    # knowledge base
    user_command: str | None
    # QA Agent
    subqueries: list[str] | None
    human_in_the_loop_decision: HITLDecision | None
    retrieved_documents: list[str] | None
    retrieved_metadatas: list[dict] | None
    summary: str | None
    # Discovery Agent
    clusters: list | None
    cluster_summaries: list[str] | None
    summary_of_summaries: str | None
