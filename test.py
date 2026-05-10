from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langfuse.langchain import CallbackHandler
from typing import TypedDict

# Langfuse tracing handler
langfuse_handler = CallbackHandler()

# Define graph state
class State(TypedDict):
    message: str
    response: str

# Single node — calls the LLM
def chat_node(state: State) -> State:
    llm = ChatOpenAI(model="gpt-4o-mini")
    result = llm.invoke(
        state["message"],
        config={"callbacks": [langfuse_handler]}
    )
    return {"response": result.content}

# Build the graph
graph = StateGraph(State)
graph.add_node("chat", chat_node)
graph.set_entry_point("chat")
graph.add_edge("chat", END)
app = graph.compile()

# Run it
output = app.invoke({"message": "Say hello in 3 words."})

print("✓ LangGraph ran successfully")
print("✓ Response:", output["response"])
print("✓ Check Langfuse dashboard for the trace → langfuse.com")