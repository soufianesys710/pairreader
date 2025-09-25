# from pairreader.agents import PairReader
from pairreader.docparser import DocParser
from pairreader.vectorestore import VectorStore
from pairreader.agents import PairReaderAgent
import logging
import chainlit as cl

logger = logging.getLogger(__name__)
docparser = DocParser()
vs = VectorStore()
pairreader = PairReaderAgent(vs)

commands = [
    {
        "id": "Use",
        "description": "Use current knowledge base",
        "icon": "/public/learning.png",
    },
    {
        "id": "Update",
        "description": "Update new knowledge base",
        "icon": "/public/writing.png",
    },
    {
        "id": "Create",
        "description": "Create a new knowledge base",
        "icon": "/public/books.png",
    }
]


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            command=command["id"],
            label=command["description"],
            icon=command["icon"],
            message="",
        )
        for command in commands
    ]


@cl.on_chat_start
async def on_chat_start():
    await cl.context.emitter.set_commands(commands)

@cl.on_message
async def on_message(msg: cl.Message):
    if msg.command == "Use":
        logger.info("Command: Use knowledge base")
        await cl.Message(content="Welcome!!").send()

    elif msg.command in ["Create", "Update"]:
        if msg.command == "Create":
            logger.info("Command: Create new knowledge base")
            logger.info("Flushing knowledge base...")
            vs.flush()
        else:
            logger.info("Command: Update knowledge base")
        files = await cl.AskFileMessage(
            content="Please upload your files to help out reading!", 
            accept=["text/plain", "application/pdf"],
            max_size_mb=10,
            max_files=5,
        ).send()
        logger.info(f"Files uploaded: {[f.name for f in files]}")
        for f in files:
            logger.info(f"Parsing file: {f.name}")
            docparser.parse(f.path)
            logger.info(f"Chunking file: {f.name}")
            chunks = docparser.get_chunks()
            logger.info(f"Ingesting chunks to the vector store, file: {f.name}")
            metadatas = [{"fname": f.name}] * len(chunks)
            vs.ingest_chunks(chunks, metadatas)
        logger.info(f"Files ready: {[f.name for f in files]}")

    else:
        final_summary_msg = cl.Message(content="")
        for aichunk, metadata in pairreader.stream({"user_query": msg.content}, stream_mode="messages"):
            if aichunk.content and metadata["langgraph_node"] == "info_summarizer":
                await final_summary_msg.stream_token(aichunk.content)
        await final_summary_msg.update()
