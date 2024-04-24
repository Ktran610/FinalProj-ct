import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_service import QueryRunner
from langchain_core.messages import HumanMessage, AIMessage
from config import configs  # Ensure this is correctly imported

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

chat_histories = {}

class ChatRequest(BaseModel):
    session_id: str
    query: str

runner = QueryRunner()
@app.post("/query")
async def query(query: str):
    try:
        result = runner.run_query(query)
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/")
async def chat(chat_request: ChatRequest):
    session_id = chat_request.session_id
    user_input = chat_request.query

    if session_id not in chat_histories:
        chat_histories[session_id] = []

    if user_input == 'exit':
        # Optionally, you might want to handle session cleanup here
        del chat_histories[session_id]  # Remove chat history
        return {"response": "Session ended"}

    # Retrieve the chat history for the current session
    chat_history = chat_histories[session_id]

    response = runner.process_chat(user_input, chat_history)

    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response))

    # Store the updated chat history back in the dictionary
    chat_histories[session_id] = chat_history

    return {"response": response}

if __name__ == "__main__":
    uvicorn.run("main:app", host=configs.host, port=configs.port, reload=True)
