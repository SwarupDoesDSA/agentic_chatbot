from agentic_chatbot_backend import chatbot
from langchain_core.messages import HumanMessage

thread_id = 1
config = {"configurable": {"thread_id": thread_id}}


while True:
    user_message = input("Type here: ").strip()

    if user_message.lower() in {"exit", "quit", "bye"}:
        break
    if not user_message:
        continue

    try:
        response = chatbot.invoke(
            {"messages": [HumanMessage(content=user_message)]},
            config=config,
        )
        print("AI:", response["messages"][-1].content)
    except Exception as exc:
        print(f"AI: I couldn't respond right now. The model is unavailable or the API key/configuration is invalid. Error: {exc}")