from langgraph.config import get_stream_writer
from langgraph.graph.state import StateGraph
from langchain_core.runnables import RunnableConfig
from functools import wraps
from typing import Dict, List, Tuple, Any, Literal, Optional, Union
import logging
import chainlit as cl

# ============================================================================
# Core Node Abstractions (Most Important)
# ============================================================================

class UserIO:
    """Handle user input/output operations, abstracting away UI framework details."""

    async def ask(self, type: Literal["text", "file"], message: str, timeout: Optional[int] = None) -> Union[str, List[Any]]:
        """
        Ask the user for input.

        Args:
            type: Type of input to request ("text" or "file")
            message: Message to display to the user
            timeout: Optional timeout in seconds

        Returns:
            For "text": The user's text response
            For "file": List of uploaded files
        """
        if type == "text":
            res = await cl.AskUserMessage(content=message, timeout=timeout).send()
            return res["output"] if res else ""
        elif type == "file":
            res = await cl.AskFileMessage(content=message, timeout=timeout, max_files=5, max_size_mb=10).send()
            return res if res else []
        else:
            raise ValueError(f"Unknown ask type: {type}")

    async def send(self, message: str, stream: bool = False):
        """
        Send a message to the user.

        Args:
            message: Message to send
            stream: Whether to stream the message (for future use)
        """
        await cl.Message(content=message).send()

    async def stream(self, llm, messages) -> str:
        """
        Stream LLM response to the user.

        Args:
            llm: The language model to stream from
            messages: The messages to send to the LLM

        Returns:
            The complete streamed content
        """
        cl_msg = cl.Message(content="")
        async for chunk in llm.astream(messages):
            if chunk.content:
                await cl_msg.stream_token(chunk.content)
        await cl_msg.update()
        return cl_msg.content


class BaseNode(UserIO):
    """
    Base class for all LangGraph nodes in PairReader.

    Provides common functionality:
    - User I/O operations (via UserIO)
    - Dynamic parameter management (set_params/get_params)
    - Standard node interface

    All nodes should inherit from this class or its subclasses.
    """

    def set_params(self, **params):
        """
        Set parameters dynamically on the node instance.

        Args:
            **params: Key-value pairs of parameters to set
        """
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def get_params(self):
        """
        Get all public parameters from the node instance.

        Returns:
            Dict of parameter names and values (excludes private attributes starting with _)
        """
        return {key: value for key, value in self.__dict__.items() if not key.startswith("_")}

    async def __call__(self, state: Dict, *args, **kwargs) -> Dict:
        """
        Execute the node logic.

        Args:
            state: The current state dictionary (typically PairReaderState)
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Dict: State updates to merge into the current state

        Note:
            Subclasses must implement this method with their specific logic.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement __call__")


class LLMNode(BaseNode):
    """
    Base class for nodes that use Language Models.

    Encapsulates the standard LLM configuration pattern with optional:
    - Fallback LLM (optional, defaults to None)
    - Tool binding (optional, for routing/function calling)
    - Structured output (optional, for Pydantic models)

    Usage:
        # Simple LLM node
        class MyNode(LLMNode):
            @Verboser(verbosity_level=2)
            async def __call__(self, state: PairReaderState) -> Dict:
                response = await self.llm.ainvoke(messages)
                return {"key": response.content}

        # With structured output
        class MyStructuredNode(LLMNode):
            def __init__(self, **kwargs):
                super().__init__(structured_output_schema=MySchema, **kwargs)

        # With tools
        class MyRouterNode(LLMNode):
            def __init__(self, **kwargs):
                super().__init__(tools=[self.tool1, self.tool2], **kwargs)
    """

    def __init__(
        self,
        llm_name: str = "anthropic:claude-3-5-haiku-latest",
        fallback_llm_name: Optional[str] = "anthropic:claude-3-7-sonnet-latest",
        tools: Optional[List[Any]] = None,
        structured_output_schema: Optional[type] = None,
        **kwargs
    ):
        """
        Initialize LLM node with optional configurations.

        Args:
            llm_name: Name of the primary language model
            fallback_llm_name: Name of the fallback language model (None to disable)
            tools: List of tools for .bind_tools() (None to disable)
            structured_output_schema: Pydantic model for structured output (None to disable)
            **kwargs: Additional parameters passed to parent classes
        """
        super().__init__(**kwargs)
        self.llm_name = llm_name
        self.fallback_llm_name = fallback_llm_name
        self.tools = tools
        self.structured_output_schema = structured_output_schema

    @property
    def llm(self):
        """
        Returns configured LLM with optional fallback, tools, and structured output.

        This is a property to ensure fresh initialization with current parameters
        whenever the LLM is accessed.

        Returns:
            Configured LLM instance
        """
        from langchain.chat_models import init_chat_model

        # Initialize primary LLM
        primary_llm = init_chat_model(self.llm_name)

        # Apply tools if provided
        if self.tools:
            primary_llm = primary_llm.bind_tools(tools=self.tools, parallel_tool_calls=False)

        # Apply structured output if provided
        if self.structured_output_schema:
            primary_llm = primary_llm.with_structured_output(self.structured_output_schema)

        # Add fallback if provided
        if self.fallback_llm_name:
            fallback_llm = init_chat_model(self.fallback_llm_name)
            if self.tools:
                fallback_llm = fallback_llm.bind_tools(tools=self.tools, parallel_tool_calls=False)
            if self.structured_output_schema:
                fallback_llm = fallback_llm.with_structured_output(self.structured_output_schema)
            primary_llm = primary_llm.with_fallbacks([fallback_llm])

        return primary_llm


class RetrievalNode(BaseNode):
    """
    Base class for nodes that interact with vector stores.

    Encapsulates the common pattern of vector store access for:
    - Document retrieval
    - Sampling and clustering
    - Metadata queries

    Usage:
        class MyRetrieverNode(RetrievalNode):
            def __init__(self, vectorstore: VectorStore, custom_param: int = 10, **kwargs):
                super().__init__(vectorstore=vectorstore, **kwargs)
                self.custom_param = custom_param

            @Verboser(verbosity_level=2)
            async def __call__(self, state: PairReaderState) -> Dict:
                results = self.vectorstore.query(...)
                return {"documents": results}
    """

    def __init__(self, vectorstore, **kwargs):
        """
        Initialize retrieval node with vector store.

        Args:
            vectorstore: VectorStore instance for document retrieval
            **kwargs: Additional parameters passed to parent classes
        """
        super().__init__(**kwargs)
        self.vectorstore = vectorstore


# ============================================================================
# Agent Abstraction
# ============================================================================

class BaseAgent:
    """Base class for LangGraph agents with common initialization and workflow patterns."""

    def __init__(self, state: type, nodes: List[Tuple[str, Any]]):
        """
        Initialize the base agent.

        Args:
            state: The state type/class for the StateGraph (e.g., PairReaderState)
            nodes: List of tuples (node_name, node_instance) representing graph nodes
        """
        self.builder = StateGraph(state)
        self.nodes = nodes

        # Register all nodes with the graph and as instance attributes
        for node in self.nodes:
            setattr(self, node[0], node[1])
            self.builder.add_node(node[0], node[1])

    async def __call__(self, input: Dict, config: RunnableConfig):
        """Execute the workflow with given input and config."""
        return await self.workflow.ainvoke(input=input, config=config)

    def set_params(self, **params):
        """Propagate parameters to all nodes that support set_params."""
        for node in self.nodes:
            node[1].set_params(**params)


# ============================================================================
# Utility Decorators
# ============================================================================

class Verboser:
	"""
	Decorator class to combine logging and streaming verbosity.

	Usage:
		@Verboser(verbosity_level=2)
		async def __call__(self, state):
			...

	Verbosity levels:
		0: No verbosity
		1: LangGraph streaming only
		2: LangGraph streaming + logging
		3: LangGraph streaming + logging with debug
	"""
	def __init__(self, verbosity_level: int = 2):
		"""
		Initialize Verboser with a verbosity level.

		Args:
			verbosity_level: Level of verbosity (0-3)
		"""
		self.verbosity_level = verbosity_level

	@staticmethod
	def logging_verbosity(func=None, *, debug=False):
		"""
		Decorator to log the start and finish of a class method using self.__class__.__name__.
		If debug=True, also logs the function's input arguments and output.
		"""
		def decorator(inner_func):
			@wraps(inner_func)
			async def wrapper(self, *args, **kwargs):
				logger = logging.getLogger(__name__)
				logger.info(f"{self.__class__.__name__} started")
				if debug:
					logger.debug(f"{self.__class__.__name__} input args: {args}, kwargs: {kwargs}")
				result = await inner_func(self, *args, **kwargs)
				if debug:
					logger.debug(f"{self.__class__.__name__} output: {result}")
				logger.info(f"{self.__class__.__name__} finished")
				return result
			return wrapper
		if func is None:
			return decorator
		else:
			return decorator(func)

	@staticmethod
	def langgraph_stream_verbosity(func):
		"""
		Decorator to call langgraph_stream_verbosity with the class name at the start of a class method.
		"""
		@wraps(func)
		async def wrapper(self, *args, **kwargs):
			get_stream_writer()(f"{self.__class__.__name__} started")
			result = await func(self, *args, **kwargs)
			get_stream_writer()(f"{self.__class__.__name__} finished")
			return result
		return wrapper

	def __call__(self, func):
		"""Apply decorators based on verbosity level."""
		if self.verbosity_level == 0:
			return func
		elif self.verbosity_level == 1:
			return self.langgraph_stream_verbosity(func)
		elif self.verbosity_level == 2:
			return self.langgraph_stream_verbosity(self.logging_verbosity(func))
		elif self.verbosity_level >= 3:
			return self.langgraph_stream_verbosity(self.logging_verbosity(func, debug=True))
		return func
