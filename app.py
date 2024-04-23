import os
from dotenv import load_dotenv
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from tools.environment_tools import DescribeEnvironment
from tools.item_tools import ListItems, StoreItem

load_dotenv()

# Retrieve Groq API key from environment variables
# groq_api_key = os.getenv('GROQ_API_KEY')
open_api_key = os.getenv('OPENAI_API_KEY')


def main():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an LLM narrating a chat based fantasy RPG (role-playing-game).  You use the provided tools to interact and get information about the user's environment. You creatively embellish the environment and interaction descriptions while conforming to the information provided by the tools. When the user asks questions unrelated to the tools suggest actions provided by the tools. When interacting with items always use the items item_id attribute. Internally the system uses the word storage, but the user maybe use different words such as backpack, inventory, pockets, etc",
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # Initialize memory and session state
    # memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history")

    # if "chat_history" not in st.session_state:
    #     st.session_state.chat_history = []

    # Initialize Groq Langchain chat object
    chat = ChatOpenAI(model="gpt-4-turbo", temperature=0)
    # model_name=model,

    # Define the tools with proper configurations
    tools = [DescribeEnvironment(), ListItems(), StoreItem()]

    # Initialize agent with the correct setup
    # agent = initialize_agent(
    #     tools=tools,
    #     llm=open_ai_chat,
    #     agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    #     verbose=True,
    #     prompt=prompt,
    #     memory=memory,
    # )
    # agent="conversational-react-description",

    agent = create_openai_tools_agent(chat, tools, prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    demo_ephemeral_chat_history = ChatMessageHistory()

    # Get user input
    #user_question = st.text_area("Ask a question:")


    #print("Ask a question")

    # Start the loop
    keep_asking = True

    while keep_asking:
        # Get the current question

        # Ask the user the question
        user_question = input(f"ADVENTURER: ")

        demo_ephemeral_chat_history.add_user_message(user_question + " environnment_id=def")

        response = agent_executor.invoke({"messages": demo_ephemeral_chat_history.messages})

        demo_ephemeral_chat_history.add_ai_message(response['output'])

        print(f"BOT: {response['output']}")



if __name__ == "__main__":
    main()
