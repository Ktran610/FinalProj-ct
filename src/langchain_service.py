from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA, ConversationalRetrievalChain, StuffDocumentsChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.vectorstores.faiss import FAISS
from config import configs
import cohere

import os
os.environ['OPENAI_API_KEY'] = configs.openai_api_key
os.environ['COHERE_API_KEY'] = configs.cohere_api_key
class QueryRunner:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = FAISS.load_local(configs.vector_store_path, self.embeddings, allow_dangerous_deserialization=True)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={'k': configs.retrieval_k})
        self.llm = ChatOpenAI(model=configs.name_model_gpt, temperature=configs.temperature)
        self.prompt_template = self.set_custom_prompt()
        self.cohere_client = cohere.Client(os.getenv('COHERE_API_KEY'))

        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": self.prompt_template},
            verbose=True,
            return_source_documents=True,
        )

        self.chain_conversation = self.create_chain()

    def run_query(self, query):
        print("query", query)

        # Retrieve documents
        retrieved_documents = self.retriever.get_relevant_documents(query)

        # Rerank documents using Cohere Reranker
        reranked_documents = self.rerank_documents(query, retrieved_documents)

        # Set context from reranked documents
        context = "\n".join([doc.page_content for doc in reranked_documents])

        # Run the QA chain
        result_query = self.qa({"query": query, "context": context})
        print(result_query)
        return result_query['result']

    def rerank_documents(self, query, documents):
        texts = [doc.page_content for doc in documents]
        rerank_response = self.cohere_client.rerank(
            model="rerank-multilingual-v3.0",
            query=query,
            documents=texts,
            top_n=len(texts),
            return_documents=True
        )
        reranked_docs = sorted(
            zip(documents, rerank_response.results),
            key=lambda x: x[1].relevance_score,
            reverse=True
        )

        return [doc for doc, result in reranked_docs]

    def set_custom_prompt(self):
        template = """Use the following information to answer the user's question.
        You are a bot serving the Duy Tan University community by providing information and answering questions in full detail.
        All your answers must be in Vietnamese.
        Context: {context}
        Question: {question}
        """
        return PromptTemplate(template=template, input_variables=['context', 'question'])


    def create_chain(self):

        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the user's questions in full detail about Duy Tan University based on the context: {context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}")
        ])

        chain = create_stuff_documents_chain(llm=self.llm, prompt=prompt)

        retriever_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
        ])

        history_aware_retriever = create_history_aware_retriever(
            llm=self.llm,
            retriever=self.retriever,
            prompt=retriever_prompt
        )

        retrieval_chain = create_retrieval_chain(
            history_aware_retriever,
            chain
        )
        return retrieval_chain

    def process_chat(self, question, chat_history):
        print("question", question)
        print("chat_history", chat_history)
        response = self.chain_conversation.invoke({
            "chat_history": chat_history,
            "input": question,
        })
        print(response)
        return response['answer']



