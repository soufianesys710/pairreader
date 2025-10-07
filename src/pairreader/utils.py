from langgraph.config import get_stream_writer
from langgraph.graph.state import StateGraph
from langchain_core.runnables import RunnableConfig
from functools import wraps
from typing import Dict, List, Tuple, Any, Literal, Optional, Union
import logging
import chainlit as cl

def logging_verbosity(func=None, *, debug=False):
	"""
	Decorator to log the start and finish of a class method using self.__class__.__name__.
	If debug=True, also logs the function's input arguments and output.
	Usage:
		@logging_verbosity
		async def ...
		@logging_verbosity(debug=True)
		async def ...
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

class ParamsMixin:
    """Set and get params, designed for langgraph graph nodes."""
    def set_params(self, **params):
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def get_params(self):
        return {key: value for key, value in self.__dict__.items() if not key.startswith("_")}


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

