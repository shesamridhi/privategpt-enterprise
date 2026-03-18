import os, tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from app.rag.vectorstore import get_vectorstore

def ingest_document(file_bytes: bytes, filename: str) -> dict:
    suffix = os.path.splitext(filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        loader = PyPDFLoader(tmp_path) if suffix == ".pdf" else TextLoader(tmp_path)
        raw_docs = loader.load()
        for doc in raw_docs:
            doc.metadata["source"] = filename
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(raw_docs)
        vs = get_vectorstore()
        vs.add_documents(chunks)
        return {"chunks": len(chunks), "filename": filename}
    finally:
        os.unlink(tmp_path)
