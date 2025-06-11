from fastapi import FastAPI, HTTPException
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from datetime import datetime, timezone
import uvicorn
from pydantic import BaseModel

app = FastAPI()

class ChatInput(BaseModel):
    input: str

def get_current_time() -> str:
    """Returns current UTC time in ISO-8601 format"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

llm = ChatOllama(model="mistral", temperature=0.5)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. Respond to time requests with:
Current UTC time: [time]
For other questions, answer normally."""),
    ("user", "{input}")
])

chain = prompt | llm | StrOutputParser()

@app.post("/chat")
async def chat_endpoint(chat_input: ChatInput):
    user_input = chat_input.input.lower()
    
    if "time" in user_input:
        return {"response": f"Current UTC time: {get_current_time()}"}
    
    try:
        response = await chain.ainvoke({"input": user_input})
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def health_check():
    return {"status": "OK", "message": "Chat API is running"}

if __name__ == "__main__":
    uvicorn.run("chatbot:app", host="0.0.0.0", port=8000, reload=True)