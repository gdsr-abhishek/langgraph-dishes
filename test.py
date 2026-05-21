from dotenv import load_dotenv
from langchain.tools import tool
load_dotenv()
from langchain.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langfuse.langchain import CallbackHandler
from langgraph.graph import MessagesState
from typing import TypedDict

# Langfuse tracing handler
langfuse_handler = CallbackHandler()
memory = MemorySaver()
# Define graph state
class State(MessagesState):
    pass
@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    # mock response for now
    return f"The weather in {city} is sunny and 28°C."
tools = [get_weather]
tools_by_name={tool.name:tool for tool in tools}
def tool_node(state:dict):
    result=[]
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation= tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation,tool_call_id=tool_call["id"]))
    return{"messages":result}
from typing import Literal
from langgraph.graph import StateGraph,START,END

def should_continue(state:MessagesState) -> Literal["tool_node","END"]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tool"
    return END
# Single node — calls the LLM
def chat_node(state: State) -> State:
    llm = ChatOpenAI(model="gpt-4o-mini")
    llm_with_tools = llm.bind_tools([get_weather])
    result = llm_with_tools.invoke(
        state["messages"]
    )
    return {"messages": [result]}

# Build the graph
graph = StateGraph(State)
graph.add_node("chat", chat_node)
graph.add_node("tool",tool_node)
graph.set_entry_point("chat")
graph.add_conditional_edges(
"chat",
should_continue,
["tool",END]
)
graph.add_edge("tool","chat")
app = graph.compile(checkpointer=memory)

config = {"callbacks": [langfuse_handler], "configurable": {"thread_id": "1"}}

app.invoke({"messages": [HumanMessage(content="Hello, I am Abhishek , I live in Vizag city")]}, config=config)
output = app.invoke({"messages": [HumanMessage(content="What is my name and also what is the current weather?")]}, config=config)

print(output["messages"][-1].content)

