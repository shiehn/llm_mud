# app.py
import os
from dotenv import load_dotenv
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from tools.environment_tools import DescribeEnvironment
from tools.item_tools import ListItems, StoreItem

load_dotenv()

# Retrieve OpenAI API key from environment variables
open_api_key = os.getenv('OPENAI_API_KEY')


class RPGChat:
    def __init__(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an LLM narrating a chat-based fantasy RPG. You use provided tools to interact with the user's environment, embellishing the environment and interactions based on tool outputs. Use item IDs when referring to items in storage.",
                ),
                MessagesPlaceholder(variable_name="messages"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        self.chat = ChatOpenAI(model="gpt-4-turbo", temperature=0)
        tools = [DescribeEnvironment(), ListItems(), StoreItem()]

        self.agent = create_openai_tools_agent(self.chat, tools, prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=tools, verbose=True)
        self.history = ChatMessageHistory()

    def ask_question(self, question):
        self.history.add_user_message(question)
        response = self.agent_executor.invoke({"messages": self.history.messages})
        self.history.add_ai_message(response["output"])
        return response["output"]


if __name__ == "__main__":
    rpg_chat = RPGChat()

    keep_asking = True

    while keep_asking:
        user_question = input("ADVENTURER: ")
        if user_question.lower() in ["quit", "exit"]:
            keep_asking = False
        else:
            response = rpg_chat.ask_question(user_question)
            print(f"BOT: {response}")
