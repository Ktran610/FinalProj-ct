from fastapi import FastAPI, HTTPException

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores.faiss import FAISS

import os
os.environ['OPENAI_API_KEY'] = ""

embeddings = OpenAIEmbeddings()
app = FastAPI()


template = """Bạn là chatbot cho trường đại học Duy Tân, bạn cần đọc và hiểu nội dung bên dưới rồi đưa ra câu trả lời.
 Nếu các câu hỏi hoặc nói về những vấn đề không liên quan, bạn chỉ cần trả lời là hiện tại dữ liệu tôi chưa được cập nhật đủ.
{context}

"""

custom_prompt_template = """Sử dụng các thông tin sau đây để trả lời câu hỏi của người dùng.
Nếu bạn không biết câu trả lời, chỉ cần nói rằng bạn không biết, đừng cố bịa ra câu trả lời.
Tất cả câu trả lời của bạn đều phải trả lời bằng tiếng việt
Context: {context}
Question: {question}
"""

from langchain import PromptTemplate
def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt


#PROMPT = PromptTemplate(template=template, input_variables=["context"])
chain_type_kwargs = {"prompt": set_custom_prompt()}
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
vectorstore = FAISS.load_local('data/vectorstore', embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={'k': 1})

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs=chain_type_kwargs,
)


@app.post("/conversation")
async def conversation(query: str):
    try:
        result = qa.run(query=query)
        return {"response": result}
    except Exception as e:
        raise HTTPException(detail=str(e), status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5566)