import dotenv
from langchain.chat_models.openai import ChatOpenAI
from langchain.agents.load_tools import load_tools
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

dotenv.load_dotenv()

tools = load_tools(["serpapi"])
llm = ChatOpenAI(temperature=0.1)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
                You are a very knowledge financial writer, your name is {name}, you will first introduce yourself, then based on information user input, 
                you will write an article contains:
                a title with no more than 55 characters, description with no more than 160 characters and body no more than 500 words
            """
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ]
)

agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

message_history = ChatMessageHistory()
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history=lambda session_id: message_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

# agent_with_chat_history.invoke({"input": "Do you know any recent events related to Fed?"}, config={"configurable": {"session_id": "<foo>"}})
