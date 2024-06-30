import dotenv
from langchain.chat_models.openai import ChatOpenAI
from langchain.agents.load_tools import load_tools
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

dotenv.load_dotenv()

tools = load_tools(["serpapi"])
llm = ChatOpenAI(
    # model_name="gpt-3.5-turbo", # Cheaper but less reliable
    model_name="gpt-4",
    temperature=0,
    max_tokens=2000
)


def agent(urls: str, title_character_count: int, desc_character_count: int, body_word_count: int):
    loader = UnstructuredURLLoader(urls=[urls])
    doc = loader.load()
    if len(doc) != 0:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60)
        doc_split = text_splitter.split_documents(doc)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                    You are a very knowledge financial writer, you will first introduce yourself, then based on information user input, 
                    you will write an article contains:
                    a title with no more than {title_character_count} characters, description with no more than {desc_character_count} characters and body no more than {body_word_count} words
                    
                    After finish writing an article you will translate it Vietnamese. Then show both versions to user.
                """
            ),
            ("user", "{doc_split}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ]
    )

    agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    message_history = ChatMessageHistory()
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        get_session_history=lambda session_id: message_history,
        input_messages_key="doc_split",
        history_messages_key="chat_history"
    )

    return agent_with_chat_history.invoke(
        {"title_character_count": title_character_count, "desc_character_count": desc_character_count,
         "body_word_count": body_word_count, "doc_split": doc_split},
        config={"configurable": {"session_id": "<foo>"}})

