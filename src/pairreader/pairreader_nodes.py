from pairreader.schemas import PairReaderState
from pairreader.vectorestore import VectorStore
from pairreader.docparser import DocParser
from pairreader.utils import Verboser, ParamsMixin, UserIO
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.types import interrupt, Command
from typing import List, Optional, Dict, Any, Annotated, Literal

class KnowledgeBaseHandler(UserIO, ParamsMixin):
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
    # @cl.step(type="ChainlitCommandHandler", name="ChainlitCommandHandler")
    async def __call__(self, state: PairReaderState, *args, **kwds):
        """Handle Chainlit commands and file uploads."""
        # if the user sends a command
        if (user_command := state.get("user_command")):
            if user_command == "Create":
                self.vectorstore.flush()
            files = await self.ask(
                type="file",
                message="Please upload your files to help out reading!",
                timeout=90
            )
            if not files:
                await self.send(
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
                await self.send(
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
                tools=[self.qa_agent, self.discovery_agent], 
                parallel_tool_calls=False
            )
            .with_fallbacks([
                init_chat_model(self.fallback_llm_name)
                .bind_tools(
                    tools=[self.qa_agent, self.discovery_agent], 
                    parallel_tool_calls=False
                )
            ])
        )
    
    @Verboser(verbosity_level=2)
    async def __call__(self, state: PairReaderState) -> Command:
        route_prompt = """
        You are a pair-reader agent that helps users chat with information from a knowledge base containing their uploaded documents.
        You have two sub-agents: QAAgent (DEFAULT) and DiscoveryAgent (SPECIAL CASES ONLY).

        **QAAgent (DEFAULT)** - Use for ALL regular questions and information requests:
        - Any question seeking specific information from the documents
        - Questions asking "what", "how", "why", "when", "where" about content
        - Requests to explain concepts, summarize specific topics, or find information
        - Examples: "What does this say about X?", "Explain Y", "How many Z are mentioned?"

        **DiscoveryAgent (SPECIAL CASES ONLY)** - Use ONLY when user explicitly requests exploration:
        - User explicitly asks for: "overview", "explore", "discover", "main themes", "main ideas", "key ideas", "overall summary"
        - User wants high-level exploration without specific questions
        - Examples: "Give me an overview", "What are the main themes?", "Explore the documents"

        IMPORTANT: Default to QAAgent unless the user explicitly uses exploration keywords.
        Most queries should go to QAAgent - it handles all regular information requests.
        """
        messages = [
            SystemMessage(content=route_prompt),
            HumanMessage(content=f"User query: {state['user_query']}")
        ]
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