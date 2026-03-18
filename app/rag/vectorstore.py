from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from app.core.config import settings

_vectorstore = None

def get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=settings.OPENAI_API_KEY
        )
        _vectorstore = Chroma(
            collection_name="privategpt_docs",
            embedding_function=embeddings,
            persist_directory="./chroma_db"
        )
    return _vectorstore
