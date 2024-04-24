# markdown_processor.py

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

class MarkdownLoader:
    def __init__(self, path):
        self.path = path
        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ]

        text_loader_kwargs = {'autodetect_encoding': True}
        self.loader = DirectoryLoader(self.path, glob="**/*.md", loader_cls=TextLoader,
                                      show_progress=True, loader_kwargs=text_loader_kwargs)
        pages = self.loader.load()
        self.docs = ' '.join(page.page_content for page in pages)

    def split_markdown(self):
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on = self.headers_to_split_on, strip_headers=False
        )

        md_header_splits= markdown_splitter.split_text(self.docs)

        chunk_size = 1000
        chunk_overlap = 30
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        # Split
        splits = text_splitter.split_documents(md_header_splits)

        return splits
    def split_markdown_2(self):
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on = self.headers_to_split_on, strip_headers=False
        )
        return markdown_splitter.split_text(self.docs)

if __name__ == "__main__":
    headers = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    processor = MarkdownLoader('../../data/demo_data/')
    md_header_splits = processor.split_markdown()
    print(len(md_header_splits))
    print(md_header_splits[-4])
