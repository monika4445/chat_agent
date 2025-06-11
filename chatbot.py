from fastapi import FastAPI
from langgraph.graph import StateGraph, END
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from typing import TypedDict, Annotated, List, Union
import operator
from datetime import datetime, timezone
import uvicorn
import subprocess
import time
from pydantic import BaseModel
from enum import Enum

# Initialize FastAPI with better docs
app = FastAPI(
    title="LangGraph Chat Agent",
    description="API for the test task with time-telling functionality",
    version="1.0.0"
)

# Pydantic models for Swagger
class MessageType(str, Enum):
    human = "human"
    ai = "ai"

class Message(BaseModel):
    content: str
    type: MessageType

class ChatRequest(BaseModel):
    messages: List[Message]

class ChatResponse(BaseModel):
    messages: List[Message]

# Ollama setup
def start_ollama():
    try:
        subprocess.run(["ollama", "list"], check=True, 
                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except:
        print("🔄 Starting Ollama...")
        subprocess.Popen(["ollama", "serve"])
        time.sleep(3)

start_ollama()

# LangGraph setup
class AgentState(TypedDict):
    messages: Annotated[
        List[Union[HumanMessage, AIMessage]], 
        operator.add
    ]

llm = ChatOllama(model="mistral", temperature=0.5)

def agent_node(state: AgentState):
    messages = state["messages"]
    last_msg = messages[-1].content.lower()
    
    if "time" in last_msg:
        return {"messages": [AIMessage(content=f"Current UTC: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")]}
    
    response = llm.invoke(messages)
    return {"messages": [response]}

workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.set_entry_point("agent")
workflow.add_edge("agent", END)
langgraph_app = workflow.compile()

# API Endpoint with proper docs
@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="Chat with the agent",
    description="Send messages to the agent and get responses. The agent can tell current UTC time.",
    tags=["Chat"]
)
async def chat_endpoint(request: ChatRequest):
    """
    Example request:
    ```json
    {
        "messages": [
            {
                "content": "What time is it?",
                "type": "human"
            }
        ]
    }
    """
    # Convert to LangChain messages
    lc_messages = []
    for msg in request.messages:
        if msg.type == MessageType.human:
            lc_messages.append(HumanMessage(content=msg.content))
        else:
            lc_messages.append(AIMessage(content=msg.content))
    
    result = await langgraph_app.ainvoke({"messages": lc_messages})
    
    # Convert back to our response format
    response_messages = []
    for msg in result["messages"]:
        if isinstance(msg, HumanMessage):
            response_messages.append(Message(content=msg.content, type=MessageType.human))
        else:
            response_messages.append(Message(content=msg.content, type=MessageType.ai))
    
    return {"messages": response_messages}

@app.get("/", include_in_schema=False)
async def health_check():
    return {"status": "OK"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)