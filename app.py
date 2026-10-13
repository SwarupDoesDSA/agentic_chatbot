from agentic_chatbot_backend import chatbot
from langchain_core.messages import BaseMessage, HumanMessage

thread_id = 1
config = {'configurable': {'thread_id': thread_id}}


while True:
    user_message = input("Type here: ").strip()

    if user_message.lower() in {"exit", "quit", "bye"}:
        break
    if not user_message:
        continue

    response = chatbot.invoke(
        {"messages": [HumanMessage(content=user_message)]},
        config=config,
    )
    print("AI:", response["messages"][-1].content)