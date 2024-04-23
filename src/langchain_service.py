from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores.faiss import FAISS
from config import configs

class QueryRunner:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = FAISS.load_local(configs.vector_store_path, self.embeddings, allow_dangerous_deserialization=True)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={'k': configs.retrieval_k})
        self.llm = ChatOpenAI(model=configs.model_name, temperature=configs.temperature)
        self.prompt_template = self.set_custom_prompt()
        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": self.prompt_template}
        )

    def run_query(self, query):
        return self.qa.run(query=query)

    def set_custom_prompt(self):
        template = """Sử dụng các thông tin sau đây để trả lời câu hỏi của người dùng.
Nếu bạn không biết câu trả lời, chỉ cần nói rằng bạn không biết, đừng cố bịa ra câu trả lời.
Tất cả câu trả lời của bạn đều phải trả lời bằng tiếng Việt.
Context: {context}
Question: {question}
"""
        return PromptTemplate(template=template, input_variables=['context', 'question'])
