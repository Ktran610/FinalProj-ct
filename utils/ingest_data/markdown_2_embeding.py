import os
from markdown_header_splitter import MarkdownLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
os.environ['OPENAI_API_KEY'] = ""

headers = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

processor = MarkdownLoader('../../data/demo_data/')
md_header_splits = processor.split_markdown()

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(md_header_splits, embeddings)

db.save_local("../../data/vectorstore")

print(f"Processed {len(md_header_splits)} documents.")