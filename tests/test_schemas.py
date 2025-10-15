"""
Unit tests for pairreader.schemas module.

Tests cover:
- HITLDecision Pydantic model
- PairReaderState TypedDict structure
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from pairreader.schemas import HITLDecision, PairReaderState

# ============================================================================
# HITLDecision Tests
# ============================================================================


class TestHITLDecision:
    """Test suite for HITLDecision Pydantic model."""

    @pytest.mark.unit
    def test_hitl_decision_to_query_optimizer(self):
        """Test HITLDecision with next_node='query_optimizer'."""
        decision = HITLDecision(next_node="query_optimizer")

        assert decision.next_node == "query_optimizer"

    @pytest.mark.unit
    def test_hitl_decision_to_info_retriever(self):
        """Test HITLDecision with next_node='info_retriever'."""
        decision = HITLDecision(next_node="info_retriever")

        assert decision.next_node == "info_retriever"

    @pytest.mark.unit
    def test_hitl_decision_invalid_node(self):
        """Test HITLDecision validation with invalid next_node."""
        with pytest.raises(ValueError):
            HITLDecision(next_node="invalid_node")

    @pytest.mark.unit
    def test_hitl_decision_serialization(self):
        """Test HITLDecision can be serialized to dict."""
        decision = HITLDecision(next_node="query_optimizer")
        decision_dict = decision.model_dump()

        assert decision_dict == {"next_node": "query_optimizer"}

    @pytest.mark.unit
    def test_hitl_decision_from_dict(self):
        """Test HITLDecision can be created from dict."""
        decision = HITLDecision(**{"next_node": "info_retriever"})

        assert decision.next_node == "info_retriever"


# ============================================================================
# PairReaderState Tests
# ============================================================================


class TestPairReaderState:
    """Test suite for PairReaderState TypedDict."""

    @pytest.mark.unit
    def test_empty_state_creation(self):
        """Test creating an empty PairReaderState."""
        state: PairReaderState = {
            "messages": [],
            "user_query": None,
            "user_command": None,
            "subqueries": None,
            "human_in_the_loop_decision": None,
            "retrieved_documents": None,
            "retrieved_metadatas": None,
            "summary": None,
            "clusters": None,
            "cluster_summaries": None,
            "summary_of_summaries": None,
        }

        assert state["messages"] == []
        assert state["user_query"] is None
        assert state["retrieved_documents"] is None

    @pytest.mark.unit
    def test_state_with_messages(self):
        """Test PairReaderState with messages."""
        messages = [HumanMessage(content="Test"), AIMessage(content="Response")]

        state: PairReaderState = {
            "messages": messages,
            "user_query": "Test query",
            "user_command": None,
            "subqueries": None,
            "human_in_the_loop_decision": None,
            "retrieved_documents": None,
            "retrieved_metadatas": None,
            "summary": None,
            "clusters": None,
            "cluster_summaries": None,
            "summary_of_summaries": None,
        }

        assert len(state["messages"]) == 2
        assert state["user_query"] == "Test query"

    @pytest.mark.unit
    def test_state_with_qa_fields(self):
        """Test PairReaderState with QA Agent fields."""
        state: PairReaderState = {
            "messages": [],
            "user_query": "What is ML?",
            "user_command": None,
            "subqueries": ["What is ML?", "Types of ML?"],
            "human_in_the_loop_decision": HITLDecision(next_node="info_retriever"),
            "retrieved_documents": ["doc1", "doc2"],
            "retrieved_metadatas": [{"filename": "file1.pdf"}, {"filename": "file2.pdf"}],
            "summary": "ML is a field of AI...",
            "clusters": None,
            "cluster_summaries": None,
            "summary_of_summaries": None,
        }

        assert state["subqueries"] == ["What is ML?", "Types of ML?"]
        assert state["human_in_the_loop_decision"].next_node == "info_retriever"
        assert len(state["retrieved_documents"]) == 2

    @pytest.mark.unit
    def test_state_with_discovery_fields(self):
        """Test PairReaderState with Discovery Agent fields."""
        state: PairReaderState = {
            "messages": [],
            "user_query": None,
            "user_command": None,
            "subqueries": None,
            "human_in_the_loop_decision": None,
            "retrieved_documents": None,
            "retrieved_metadatas": None,
            "summary": None,
            "clusters": [["doc1", "doc2"], ["doc3", "doc4"]],
            "cluster_summaries": ["Summary of cluster 1", "Summary of cluster 2"],
            "summary_of_summaries": "Overall summary",
        }

        assert len(state["clusters"]) == 2
        assert len(state["cluster_summaries"]) == 2
        assert state["summary_of_summaries"] == "Overall summary"

    @pytest.mark.unit
    def test_state_partial_update(self):
        """Test updating PairReaderState with partial dict."""
        initial_state: PairReaderState = {
            "messages": [],
            "user_query": "Initial query",
            "user_command": None,
            "subqueries": None,
            "human_in_the_loop_decision": None,
            "retrieved_documents": None,
            "retrieved_metadatas": None,
            "summary": None,
            "clusters": None,
            "cluster_summaries": None,
            "summary_of_summaries": None,
        }

        # Simulate partial update (as would happen in LangGraph)
        update = {"subqueries": ["query1", "query2"]}
        updated_state = {**initial_state, **update}

        assert updated_state["user_query"] == "Initial query"
        assert updated_state["subqueries"] == ["query1", "query2"]

    @pytest.mark.unit
    def test_state_with_command(self):
        """Test PairReaderState with user commands."""
        state: PairReaderState = {
            "messages": [],
            "user_query": None,
            "user_command": "Create",
            "subqueries": None,
            "human_in_the_loop_decision": None,
            "retrieved_documents": None,
            "retrieved_metadatas": None,
            "summary": None,
            "clusters": None,
            "cluster_summaries": None,
            "summary_of_summaries": None,
        }

        assert state["user_command"] == "Create"

        # Test with Update command
        state["user_command"] = "Update"
        assert state["user_command"] == "Update"
