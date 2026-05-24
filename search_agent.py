from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, MessagesState,END
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langfuse.langchain import CallbackHandler
from pydantic import BaseModel
from typing import List
import instructor
from openai import OpenAI
load_dotenv()

class ResearchSummary(BaseModel):
    topic: str
    summary: str
    key_points: List[str]
    sources: List[str]
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
client = instructor.from_openai(
    OpenAI()
)
def llm_node(state:MessagesState):

    messages = [system_prompt] + state['messages']
    response = llm_with_tools.invoke(messages)
    return {"messages":[response]}
def should_continue(state:MessagesState):
    if state["messages"][-1].tool_calls:
        return "tool"
    return "summarise"
def synthesis_node(state: MessagesState) -> dict:
    # All tool results are now in state["messages"]
    # Pass them to Instructor for structured output
    result = client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=ResearchSummary,
        messages=[
            {"role": "system", "content": "Synthesize the research into structured output"},
            {"role": "user", "content": str(state["messages"])}
        ]
    )
    return {"messages": [AIMessage(content=result.model_dump_json())]}
agent_executor =  StateGraph(MessagesState)
agent_executor.add_node("tool",tool_node)
agent_executor.add_node("llm",llm_node)
agent_executor.set_entry_point("llm")
agent_executor.add_node("summarise",synthesis_node)
agent_executor.add_conditional_edges("llm",should_continue,["tool","summarise"])
agent_executor.add_edge("summarise",END)
agent_executor.add_edge("tool","llm")
agent =agent_executor.compile(checkpointer=memory)

while True:
    query = input('Your query:')
    if query.lower() in 'stopquitexitnone':
        break
    else:
        response = agent.invoke({"messages":[HumanMessage(query)]},config=config)
        print(response["messages"][-1])