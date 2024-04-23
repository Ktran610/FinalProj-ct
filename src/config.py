from pydantic_settings import BaseSettings

class Config(BaseSettings):
    openai_api_key: str = ""
    model_name: str = "gpt-3.5-turbo-0125"
    temperature: float = 0.0
    vector_store_path: str = '../data/vectorstore'
    retrieval_k: int = 1
    host: str = '0.0.0.0'  # Default host
    port: int = 5566         # Default port

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

configs = Config()
