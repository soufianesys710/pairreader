from pairreader.schemas import PairReaderState
from pairreader.vectorestore import VectorStore
from pairreader.docparser import DocParser
from pairreader.utils import Verboser, BaseNode, LLMNode, RetrievalNode
from pairreader.prompts_msgs import PAIRREADER_PROMPTS, PAIRREADER_MSGS
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.types import interrupt, Command
from typing import List, Optional, Dict, Any, Annotated, Literal

class KnowledgeBaseHandler(BaseNode):
    """
    Handles knowledge base commands (Create, Update) and file upload logic.

    - Updates state with user_command and processes file ingestion.
    - Prompts user to upload files and ingests them into the vector store.
    - Interrupts flow if no files are uploaded within timeout.
    """
    def __init__(self, docparser: DocParser, vectorstore: VectorStore):
        self.docparser = docparser
        self.vectorstore = vectorstore

    @Verboser(verbosity_level=2)
    async def __call__(self, state: PairReaderState, *args, **kwds):
        """Handle Chainlit commands and file uploads."""
        # if the user sends a command
        if (user_command := state.get("user_command")):
            if user_command == "Create":
                await self.send(PAIRREADER_MSGS["kb_flushing"])
                self.vectorstore.flush()
            files = await self.ask(
                type="file",
                message=PAIRREADER_MSGS["kb_upload_files"],
                timeout=90
            )
            if not files:
                await self.send(
                    PAIRREADER_MSGS["kb_timeout"].format(user_command=user_command)
                )
                interrupt("")
            else:
                await self.send(PAIRREADER_MSGS["kb_processing"].format(n_files=len(files)))
                for f in files:
                    await self.send(PAIRREADER_MSGS["kb_parsing"].format(file_name=f.name))
                    self.docparser.parse(f.path)
                    chunks = self.docparser.get_chunks()
                    metadatas = [{"fname": f.name}] * len(chunks)
                    await self.send(PAIRREADER_MSGS["kb_ingesting"].format(n_chunks=len(chunks), file_name=f.name))
                    self.vectorstore.ingest_chunks(chunks, metadatas)
                # files uploaded and parsed, ask for a user query
                len_docs = self.vectorstore.get_len_docs()
                await self.send(
                    PAIRREADER_MSGS["kb_success"].format(
                        file_names=[f.name for f in files],
                        len_docs=len_docs
                    )
                )
                interrupt("")
        # the user doesn't send a command, rather he should've sent a message, don't update the state
        else:
            return {}


class QADiscoveryRouter(LLMNode):
    """
    Routes user queries to appropriate agent (QA or Discovery).

    - Uses LLM with tool binding to decide routing
    - Returns Command to navigate to selected agent
    """

    def __init__(self, **kwargs):
        """Initialize router with qa_agent and discovery_agent tools."""
        super().__init__(tools=[self.qa_agent, self.discovery_agent], **kwargs)

    @Verboser(verbosity_level=2)
    async def __call__(self, state: PairReaderState) -> Command:
        prompt = PAIRREADER_PROMPTS["qa_discovery_router"].format(user_query=state['user_query'])
        messages = [HumanMessage(content=prompt)]
        response = await self.llm.ainvoke(messages)
        tool_call = response.tool_calls[0]
        return Command(
            goto=tool_call["name"],
            update={"messages": [AIMessage(content=f"Routing to {tool_call['name']}")]}
        )


    @tool(description="Handoff to QAAgent - Use for ALL regular questions and information requests (DEFAULT)")
    def qa_agent():
        """Use this to handoff to QA agent for regular questions. This is the DEFAULT agent."""
        return "qa_agent"

    @tool(description="Handoff to DiscoveryAgent - Use ONLY when user explicitly requests overview/themes/exploration")
    def discovery_agent():
        """Use this to handoff to discovery agent ONLY for explicit exploration requests (overview, themes, key ideas, etc.)."""
        return "discovery_agent"