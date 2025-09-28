from pairreader.agents import PairReaderAgent
import chainlit as cl


pairreader = PairReaderAgent()


commands = [
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
    state = {
        "chainlit_command": msg.command if hasattr(msg, "command") else None,
        "user_query": msg.content,
    }
    final_summary_msg = cl.Message(content="")
    async for mode, data  in pairreader.workflow.astream(state, stream_mode=["messages", "updates"]):
        if mode == "updates":
            if "__interrupt__" in data:
                await cl.Message(content=data["__interrupt__"][0].value).send()
        elif mode == "messages":
            aichunk, metadata = data
            if aichunk.content and metadata.get("langgraph_node") == "info_summarizer":
                await final_summary_msg.stream_token(aichunk.content)
    await final_summary_msg.update()
