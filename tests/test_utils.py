"""
Unit tests for pairreader.utils module.

Tests cover:
- BaseNode class and its methods (set_params, get_params)
- LLMNode class and LLM property
- RetrievalNode class
- Verboser decorator
"""

import logging
from unittest.mock import AsyncMock, Mock, patch

import pytest

from pairreader.utils import BaseAgent, BaseNode, LLMNode, RetrievalNode, Verboser

# ============================================================================
# BaseNode Tests
# ============================================================================


class TestBaseNode:
    """Test suite for BaseNode class."""

    @pytest.mark.unit
    def test_base_node_initialization(self):
        """Test that BaseNode can be instantiated."""
        node = BaseNode()
        assert isinstance(node, BaseNode)

    @pytest.mark.unit
    def test_set_params(self):
        """Test that set_params correctly sets attributes."""
        node = BaseNode()
        node.test_param = "original"
        node.another_param = 10

        node.set_params(test_param="updated", another_param=20)

        assert node.test_param == "updated"
        assert node.another_param == 20

    @pytest.mark.unit
    def test_set_params_only_existing_attributes(self):
        """Test that set_params only updates existing attributes."""
        node = BaseNode()
        node.existing = "value"

        # This should not raise an error, but won't create new_param
        node.set_params(existing="updated", new_param="should_not_exist")

        assert node.existing == "updated"
        assert not hasattr(node, "new_param")

    @pytest.mark.unit
    def test_get_params(self):
        """Test that get_params returns all public parameters."""
        node = BaseNode()
        node.public_param = "public"
        node._private_param = "private"
        node.another_public = 42

        params = node.get_params()

        assert "public_param" in params
        assert "another_public" in params
        assert "_private_param" not in params
        assert params["public_param"] == "public"
        assert params["another_public"] == 42

    @pytest.mark.unit
    async def test_call_not_implemented(self):
        """Test that calling BaseNode raises NotImplementedError."""
        node = BaseNode()

        with pytest.raises(NotImplementedError, match="BaseNode must implement __call__"):
            await node({})


# ============================================================================
# LLMNode Tests
# ============================================================================


class TestLLMNode:
    """Test suite for LLMNode class."""

    @pytest.mark.unit
    def test_llm_node_initialization_defaults(self):
        """Test LLMNode initialization with default parameters."""
        node = LLMNode()

        assert node.llm_name == "anthropic:claude-3-5-haiku-latest"
        assert node.fallback_llm_name == "anthropic:claude-3-7-sonnet-latest"
        assert node.tools is None
        assert node.structured_output_schema is None

    @pytest.mark.unit
    def test_llm_node_initialization_custom(self):
        """Test LLMNode initialization with custom parameters."""
        node = LLMNode(
            llm_name="custom-model",
            fallback_llm_name="custom-fallback",
            tools=["tool1", "tool2"],
            structured_output_schema=dict,
        )

        assert node.llm_name == "custom-model"
        assert node.fallback_llm_name == "custom-fallback"
        assert node.tools == ["tool1", "tool2"]
        assert node.structured_output_schema == dict

    @pytest.mark.unit
    def test_llm_node_no_fallback(self):
        """Test LLMNode with fallback disabled."""
        node = LLMNode(fallback_llm_name=None)

        assert node.fallback_llm_name is None

    @pytest.mark.unit
    def test_llm_property_basic(self, mock_init_chat_model):
        """Test that llm property returns configured LLM."""
        node = LLMNode()

        llm = node.llm

        assert llm is not None
        # Verify that init_chat_model was called
        # (mock_init_chat_model is already patched in fixture)

    @pytest.mark.unit
    def test_llm_property_with_tools(self, mock_init_chat_model, mock_llm):
        """Test that llm property correctly binds tools."""
        tools = [Mock(), Mock()]
        node = LLMNode(tools=tools)

        llm = node.llm

        # Verify bind_tools was called (called for both primary and fallback LLM)
        assert mock_llm.bind_tools.called
        assert mock_llm.bind_tools.call_count >= 1

    @pytest.mark.unit
    def test_llm_property_with_structured_output(self, mock_init_chat_model, mock_llm):
        """Test that llm property correctly applies structured output."""
        from pydantic import BaseModel

        class TestSchema(BaseModel):
            field: str

        node = LLMNode(structured_output_schema=TestSchema)

        llm = node.llm

        # Verify with_structured_output was called (called for both primary and fallback LLM)
        assert mock_llm.with_structured_output.called
        assert mock_llm.with_structured_output.call_count >= 1

    @pytest.mark.unit
    def test_llm_property_with_fallback(self, mock_init_chat_model, mock_llm):
        """Test that llm property correctly adds fallback."""
        node = LLMNode(fallback_llm_name="fallback-model")

        llm = node.llm

        # Verify with_fallbacks was called
        mock_llm.with_fallbacks.assert_called_once()

    @pytest.mark.unit
    def test_llm_property_dynamic_update(self, mock_init_chat_model):
        """Test that llm property reflects updated parameters."""
        node = LLMNode()

        # Change parameters
        node.llm_name = "new-model"

        # Access llm property again (should reinitialize with new params)
        llm = node.llm

        assert llm is not None


# ============================================================================
# RetrievalNode Tests
# ============================================================================


class TestRetrievalNode:
    """Test suite for RetrievalNode class."""

    @pytest.mark.unit
    def test_retrieval_node_initialization(self, mock_vectorstore):
        """Test RetrievalNode initialization with vectorstore."""
        node = RetrievalNode(vectorstore=mock_vectorstore)

        assert node.vectorstore is mock_vectorstore

    @pytest.mark.unit
    def test_retrieval_node_custom_params(self, mock_vectorstore):
        """Test RetrievalNode with additional custom parameters."""
        node = RetrievalNode(vectorstore=mock_vectorstore)
        node.custom_param = 42

        assert node.vectorstore is mock_vectorstore
        assert node.custom_param == 42

    @pytest.mark.unit
    async def test_retrieval_node_call_not_implemented(self, mock_vectorstore):
        """Test that calling RetrievalNode raises NotImplementedError."""
        node = RetrievalNode(vectorstore=mock_vectorstore)

        with pytest.raises(NotImplementedError):
            await node({})


# ============================================================================
# Verboser Decorator Tests
# ============================================================================


class TestVerboser:
    """Test suite for Verboser decorator."""

    @pytest.mark.unit
    async def test_verboser_level_0(self):
        """Test Verboser with level 0 (no verbosity)."""

        class TestNode(BaseNode):
            @Verboser(verbosity_level=0)
            async def __call__(self, state):
                return {"result": "success"}

        node = TestNode()
        result = await node({})

        assert result == {"result": "success"}

    @pytest.mark.unit
    async def test_verboser_level_1(self):
        """Test Verboser with level 1 (LangGraph streaming only)."""
        with patch("pairreader.utils.get_stream_writer") as mock_stream:
            mock_writer = Mock()
            mock_stream.return_value = mock_writer

            class TestNode(BaseNode):
                @Verboser(verbosity_level=1)
                async def __call__(self, state):
                    return {"result": "success"}

            node = TestNode()
            result = await node({})

            assert result == {"result": "success"}
            # Verify stream writer was called for start and finish
            assert mock_writer.call_count == 2

    @pytest.mark.unit
    async def test_verboser_level_2(self, caplog):
        """Test Verboser with level 2 (streaming + logging)."""
        with patch("pairreader.utils.get_stream_writer") as mock_stream:
            mock_writer = Mock()
            mock_stream.return_value = mock_writer

            class TestNode(BaseNode):
                @Verboser(verbosity_level=2)
                async def __call__(self, state):
                    return {"result": "success"}

            with caplog.at_level(logging.INFO):
                node = TestNode()
                result = await node({})

            assert result == {"result": "success"}
            # Check that logging occurred
            assert any("TestNode started" in record.message for record in caplog.records)
            assert any("TestNode finished" in record.message for record in caplog.records)

    @pytest.mark.unit
    async def test_verboser_level_3(self, caplog):
        """Test Verboser with level 3 (streaming + debug logging)."""
        with patch("pairreader.utils.get_stream_writer") as mock_stream:
            mock_writer = Mock()
            mock_stream.return_value = mock_writer

            class TestNode(BaseNode):
                @Verboser(verbosity_level=3)
                async def __call__(self, state):
                    return {"result": "success"}

            with caplog.at_level(logging.DEBUG):
                node = TestNode()
                result = await node({"input": "test"})

            assert result == {"result": "success"}
            # Check that debug logging occurred
            assert any("TestNode started" in record.message for record in caplog.records)


# ============================================================================
# BaseAgent Tests
# ============================================================================


class TestBaseAgent:
    """Test suite for BaseAgent class."""

    @pytest.mark.unit
    def test_base_agent_initialization(self, empty_state):
        """Test BaseAgent initialization with nodes."""
        from pairreader.schemas import PairReaderState

        # Create mock nodes
        node1 = BaseNode()
        node2 = BaseNode()
        nodes = [("node1", node1), ("node2", node2)]

        agent = BaseAgent(state=PairReaderState, nodes=nodes)

        assert agent.nodes == nodes
        assert hasattr(agent, "node1")
        assert hasattr(agent, "node2")
        assert agent.node1 is node1
        assert agent.node2 is node2

    @pytest.mark.unit
    def test_base_agent_set_params(self):
        """Test that BaseAgent.set_params propagates to all nodes."""
        from pairreader.schemas import PairReaderState

        # Create nodes with parameters
        node1 = BaseNode()
        node1.param1 = "original1"

        node2 = BaseNode()
        node2.param1 = "original2"

        nodes = [("node1", node1), ("node2", node2)]

        agent = BaseAgent(state=PairReaderState, nodes=nodes)
        agent.set_params(param1="updated")

        assert node1.param1 == "updated"
        assert node2.param1 == "updated"

    @pytest.mark.unit
    async def test_base_agent_call(self):
        """Test BaseAgent.__call__ method."""
        from pairreader.schemas import PairReaderState

        node = BaseNode()
        nodes = [("test_node", node)]

        agent = BaseAgent(state=PairReaderState, nodes=nodes)

        # Mock the workflow
        mock_workflow = AsyncMock()
        mock_workflow.ainvoke = AsyncMock(return_value={"result": "success"})
        agent.workflow = mock_workflow

        result = await agent(input={"test": "input"}, config={"configurable": {"thread_id": "123"}})

        assert result == {"result": "success"}
        mock_workflow.ainvoke.assert_called_once()
