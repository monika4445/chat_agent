from langgraph.graph import StateGraph, END
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from typing import TypedDict, Annotated, List, Union
import operator
from datetime import datetime, timezone
import subprocess
import time

def start_ollama():
    try:
        subprocess.run(["ollama", "list"], check=True,
                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except:
        print("🔄 Starting Ollama in background...")
        subprocess.Popen(["ollama", "serve"])
        time.sleep(3)

start_ollama()

class AgentState(TypedDict):
    messages: Annotated[
        List[Union[HumanMessage, AIMessage]],
        operator.add
    ]

llm = ChatOllama(
    model="mistral",
    temperature=0.5,
    base_url="http://localhost:11434"
)

def get_current_time():
    return {"utc": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}

def agent_node(state: AgentState):
    messages = state["messages"]
    last_msg = messages[-1].content.lower()
    
    if "time" in last_msg:
        return {"messages": [AIMessage(content=f"Current UTC: {get_current_time()['utc']}")]}
    
    response = llm.invoke(messages)
    return {"messages": [response]}

workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.set_entry_point("agent")
workflow.add_edge("agent", END)

app = workflow.compile()