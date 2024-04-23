import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_service import QueryRunner
from config import configs  # Ensure this is correctly imported

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

runner = QueryRunner()

class Query(BaseModel):
    text: str
@app.post("/conversation")
async def conversation(query: Query):
    try:
        result = runner.run_query(query.text)
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host=configs.host, port=configs.port, reload=True)
