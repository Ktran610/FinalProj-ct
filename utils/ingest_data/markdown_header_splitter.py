# markdown_processor.py

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter

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
        self.docs = self.loader.load()[0].page_content

    def split_markdown(self):
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

    processor = MarkdownLoader('data/test_one_file/')
    md_header_splits = processor.split_markdown()
    print(md_header_splits)
