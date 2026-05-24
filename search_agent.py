from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage 
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, MessagesState,END
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langfuse.langchain import CallbackHandler
load_dotenv()


langfuse_handler = CallbackHandler()
memory = MemorySaver()
config = {
    "configurable": {"thread_id": "1"},
    "callbacks": [langfuse_handler]
}
system_prompt = SystemMessage(content="""You are a helpful research assistant. 
When you have search results, synthesise them into a clear, concise answer.
Always cite your sources at the end with the URL.""")
llm = ChatOpenAI(model="gpt-4o-mini")
tools = [TavilySearch(max_results=3)]
llm_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools)
def llm_node(state:MessagesState):

    messages = [system_prompt] + state['messages']
    response = llm_with_tools.invoke(messages)
    return {"messages":[response]}
def should_continue(state:MessagesState):
    if state["messages"][-1].tool_calls:
        return "tool"
    return END
agent_executor =  StateGraph(MessagesState)
agent_executor.add_node("tool",tool_node)
agent_executor.add_node("llm",llm_node)
agent_executor.set_entry_point("llm")
agent_executor.add_conditional_edges("llm",should_continue,["tool",END])
agent_executor.add_edge("tool","llm")
agent =agent_executor.compile(checkpointer=memory)

while True:
    query = input('Your query:')
    if query.lower() in 'stopquitexitnone':
        break
    else:
        response = agent.invoke({"messages":[HumanMessage(query)]},config=config)
        print(response["messages"][-1].content)