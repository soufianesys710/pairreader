from pairreader.schemas import PairReaderState
from pairreader.vectorestore import VectorStore
from pairreader.docparser import DocParser
from pairreader.utils import logging_verbosity, langgraph_stream_verbosity, ParamsMixin
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool, InjectedToolCallId, InjectedToolArg
from langgraph.prebuilt import InjectedState
from langgraph.types import interrupt, Command
from typing import List, Optional, Dict, Any, Annotated, Literal
import chainlit as cl

class KnowledgeBaseHandler(ParamsMixin):
    """
    Handles knowledge base commands (Create, Update) and file upload logic.

    - Updates state with user_command and processes file ingestion.
    - Prompts user to upload files and ingests them into the vector store.
    - Interrupts flow if no files are uploaded within timeout.
    """
    def __init__(self, docparser: DocParser, vectorstore: VectorStore):
        self.docparser = docparser
        self.vectorstore = vectorstore

    @logging_verbosity
    @langgraph_stream_verbosity
    # @cl.step(type="ChainlitCommandHandler", name="ChainlitCommandHandler")
    async def __call__(self, state: PairReaderState, *args, **kwds):
        """Handle Chainlit commands and file uploads."""
        # if the user sends a command
        if (user_command := state.get("user_command")):
            if user_command == "Create":
                self.vectorstore.flush()
            files = await cl.AskFileMessage(
                content="Please upload your files to help out reading!",
                accept=["text/plain", "application/pdf"],
                max_size_mb=10,
                max_files=5,
                timeout=90
            ).send()
            if files is None:
                await cl.Message(
                    f"You haven't uploaded any files in the 60s following your {user_command} command!"
                    "You can continue to use the your current knowledge base, or resend a Create or Update command described in the toolbox"
                )
                interrupt()
            else:
                for f in files:
                    self.docparser.parse(f.path)
                    chunks = self.docparser.get_chunks()
                    metadatas = [{"fname": f.name}] * len(chunks)
                    self.vectorstore.ingest_chunks(chunks, metadatas)
                # files uploaded and parsed, ask for a user query
                len_docs = self.vectorstore.get_len_docs()
                await cl.Message(
                    f"Files uploaded: {[f.name for f in files]}. Knowledge base now contains {len_docs} document chunks. What do you want to know?"
                )
                interrupt()
        # the user doesn't send a command, rather he should've sent a message, don't update the state
        else:
            return {}


class QADiscoveryRouter(ParamsMixin):
    def __init__(self,
        llm_name: str = "anthropic:claude-3-5-haiku-latest",
        fallback_llm_name: str = "anthropic:claude-3-7-sonnet-latest"
    ):
        self.llm_name = llm_name
        self.fallback_llm_name = fallback_llm_name

    @property
    def llm(self):
        return (
            init_chat_model(self.llm_name)
            .bind_tools(
                tools=[self.qa_agent_handoff, self.discovery_agent_handoff], 
                parallel_tool_calls=False
            )
            .with_fallbacks([
                init_chat_model(self.fallback_llm_name)
                .bind_tools(
                    tools=[self.qa_agent_handoff, self.discovery_agent_handoff], 
                    parallel_tool_calls=False
                )
            ])
        )
    
    @logging_verbosity
    @langgraph_stream_verbosity
    async def __call__(self, state: PairReaderState) -> Command[Literal["qa_agent", "discovery_agent"]]:
        route_prompt = """
        You are a pair-reader agent that helps the user chat with information from a knowledge base.
        You have two sub-agents: a QAAgent and a DiscoveryAgent.
        The QAAgent is able to answer specific questions based on available information in the knowledge base. Basically when the user knows what he's looking for.
        The DiscoveryAgent is able to help the user discover information in the knowledge base, provide overview, summary, etc. Basically when the user doesn't know what he's looking for.
        You are given a user query. You have to decide whether to route the query to the QAAgent or the DiscoveryAgent, then the sub-agents take over.
        """
        messages = [
            SystemMessage(content=route_prompt),
            HumanMessage(content=f"User query: {state['user_query']}")
        ]
        response = await self.llm.ainvoke(messages)
        return response


    @tool(description="Handoff to QAAgent")
    def qa_agent_handoff(
        state: Annotated[PairReaderState, InjectedState], 
        tool_call_id: Annotated[str, InjectedToolCallId]
    ) -> Command:
        """Use this to handoff to QA agent."""
        tool_message = ToolMessage(
            content="Handoff to QAAgent",
            name="qa_agent_handoff",
            tool_call_id=tool_call_id,
        )
        return Command(
            goto="qa_agent",  
            update={"messages": [tool_message]},
        )

    @tool(description="Handoff to DiscoveryAgent")
    def discovery_agent_handoff(
        state: Annotated[PairReaderState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId]
    ) -> Command:
        """Use this to handoff to discovery agent."""
        tool_message = ToolMessage(
            content="Handoff to DiscoveryAgent",
            name="discovery_agent_handoff",
            tool_call_id=tool_call_id,
        )
        return Command(
            goto="discovery_agent",  
            update={"messages": [tool_message]},
        )