from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv 
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

load_dotenv()

model = ChatGoogleGenerativeAI(
     model="gemini-3.8-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

class ChatState(TypedDict):

    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']

    ## Invoke llm
    response =  model.invoke(messages)

    # response store state
    return {'messages': [response]}

checkpoint = MemorySaver()

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpoint)
