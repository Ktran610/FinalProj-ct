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
            chain_type_kwargs={"prompt": self.prompt_template},
            callbacks=None,
            verbose=True,
            return_source_documents=True,
        )

        self.chain_conversation = self.create_chain()

    def run_query(self, query):
        result_query = self.qa({"query": query})
        print(result_query)
        return result_query['result']

    def set_custom_prompt(self):
        template = """Use the following information to answer the user's question.
        If you do not know the answer, just say that you do not know; do not make up an answer.
        All your answers must be in Vietnamese.
        Context: {context}
        Question: {question}
        """
        return PromptTemplate(template=template, input_variables=['context', 'question'])


    def create_chain(self):

        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the user's questions based on the context: {context}"),
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
        return response['answer']



