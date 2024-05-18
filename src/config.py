from pydantic_settings import BaseSettings

class Config(BaseSettings):
    openai_api_key: str = ""
    cohere_api_key: str = ""
    name_model_gpt: str = "gpt-4o"
    temperature: float = 0.7
    vector_store_path: str = '../data/vectorstore'
    retrieval_k: int = 6
    host: str = '0.0.0.0'  # Default host
    port: int = 5566         # Default port

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

configs = Config()
