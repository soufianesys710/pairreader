from pairreader.agents import PairReaderAgent
from pairreader.clmemory import InMemoryDataLayer
from chainlit.input_widget import Select, Switch, Slider
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


@cl.password_auth_callback
def password_auth_callback(username: str, password: str):
    # TODO: user hashed password is to be fetched from database later
    # TODO: other options can such as OAuth and Header based authentication can be explored in the cl docs 
    # CHAINLIT_AUTH_SECRET has to be set in the env variables, run `chainlit create-secret` cli to get one.
    if (username, password) == ("admin", "admin"):
        return cl.User(
            identifier="admin", metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None


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
    current_user = cl.user_session.get("user")
    await cl.context.emitter.set_commands(commands)
    settings = await cl.ChatSettings(
        [
            Select(
                id="LLM",
                label="LLM",
                values=[
                    "anthropic:claude-3-5-haiku-latest",
                    "anthropic:claude-3-7-sonnet-latest"
                ],
                initial_index=0,
            ),
            Select(
                id="Fallback LLM",
                label="Fallback LLM",
                values=[
                    "anthropic:claude-3-5-haiku-latest",
                    "anthropic:claude-3-7-sonnet-latest"
                ],
                initial_index=1,
            ),
            Switch(
                id="query_decomposition",
                label="Decompose the user query before querying the Knowledge base",
                initial=True
            ),
            Slider(
                id="n_documents",
                label="Number of documents to retrieve from the Knowledge base",
                initial=10,
                min=5,
                max=20,
                step=1,
            ),
        ]
    ).send()


@cl.on_chat_resume
async def on_chat_resume(thread):
    pass


@cl.on_chat_end
async def on_chat_end():
    pass


@cl.on_settings_update
async def on_settings_update(settings):
    pairreader.set_params(**settings)


@cl.data_layer
def get_data_layer():
    return InMemoryDataLayer()


@cl.on_message
async def on_message(msg: cl.Message):
    thread_id = cl.context.session.thread_id
    config = {"configurable": {"thread_id": thread_id}}
    input = {
        "chainlit_command": msg.command if hasattr(msg, "command") else None,
        "user_query": msg.content,
    }
    _ = await pairreader(input=input, config=config)
    # final_summary_msg = cl.Message(content="")
    # async for mode, data in pairreader.workflow.astream(
    #     state, 
    #     stream_mode=["messages", "updates"],
    #     config=config,
    # ):
    #     if mode == "updates":
    #         if "__interrupt__" in data:
    #             await cl.Message(content=data["__interrupt__"][0].value).send()
    #     elif mode == "messages":
    #         aichunk, metadata = data
    #         if aichunk.content and metadata.get("langgraph_node") == "info_summarizer":
    #             await final_summary_msg.stream_token(aichunk.content)
    # await final_summary_msg.update()


if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)