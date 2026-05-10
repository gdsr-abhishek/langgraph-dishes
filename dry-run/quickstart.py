from langchain.tools import tool
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()
model = init_chat_model(
    "openai:gpt-4o-mini",
    temperature=0
)

@tool
def multiply(a:int , b:int) -> int:
    """ Multiply 'a' and 'b'
    Args:
    a: First int
    b: Second int
    """
    return a*b
@tool
def add(a:int , b:int) -> int:
    """ add 'a' and 'b'
    Args:
    a: First int
    b: Second int
    """
    return a+b
@tool
def divide(a:int , b:int) -> float:
    """ divide 'a' and 'b'
    Args:
    a: First int
    b: Second int
    """
    return a/b
@tool
def subtract(a:int , b:int) -> int:
    """ subtract 'b' from 'a'
    Args:
    a: First int
    b: Second int
    """
    return a - b

tools = [add,multiply,divide,subtract]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)

from langchain.messages import AnyMessage
from typing_extensions import TypedDict,Annotated
import operator

class MessageState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls:int

from langchain.messages import SystemMessage

def llm_call(state:dict):
    """LLM call whether to call a tool or not"""

    return{
        "messages":[
            model_with_tools.invoke(
          [  SystemMessage(content="You are a helpful assistant who is specialized in performing arithmetic operations like addition,subtraction , multiplication and division on two numbers")

        ]
        + state["messages"]
        )
        ], "llm_calls":state.get("llm_calls",0 ) + 1
    }


from langchain.messages import ToolMessage

def tool_node(state:dict):
    result=[]
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation= tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation,tool_call_id=tool_call["id"]))
    return{"messages":result}
from typing import Literal
from langgraph.graph import StateGraph,START,END

def should_continue(state:MessageState) -> Literal["tool_node","END"]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tool_node"
    return END
agent_builder = StateGraph(MessageState)
# agent_builder.add_node("START",START)
# agent_builder.add_node("END",END)

agent_builder.add_node("llm_call",llm_call)
agent_builder.add_node("tool_node",tool_node)
agent_builder.set_entry_point("llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node",END]
    )
agent_builder.add_edge("tool_node","llm_call")
agent  = agent_builder.compile()

from IPython.display import Image,display
display(Image(agent.get_graph(xray=True).draw_mermaid_png()))

# from langchain.messages import HumanMessage
# messages = [HumanMessage(content="Add 3 and 10")]
# messages =agent.invoke({"messages":messages})
# for m in messages["messages"]:
#     m.pretty_print()

from langchain.messages import HumanMessage
messages = [HumanMessage(content="What is 144 divided by 12?")]
messages =agent.invoke({"messages":messages})
for m in messages["messages"]:
    m.pretty_print()