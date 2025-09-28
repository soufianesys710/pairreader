from langgraph.config import get_stream_writer
from functools import wraps
import logging

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
		get_stream_writer()(f"self.__class__.__name__ started")
		result = await func(self, *args, **kwargs)
		get_stream_writer()(f"self.__class__.__name__ finished")
		return result
	return wrapper
