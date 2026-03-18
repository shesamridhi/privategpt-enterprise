from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.rag.ingestor import ingest_document
from app.rag.chain import get_rag_chain
from app.api.schemas import QueryRequest, QueryResponse
import uvicorn

app = FastAPI(
    title="PrivateGPT Enterprise",
    description="Production RAG system — chat with your documents, source-cited, zero hallucination.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "online", "service": "PrivateGPT Enterprise"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith((".pdf", ".txt", ".docx")):
        raise HTTPException(status_code=400, detail="Only PDF, TXT, DOCX supported.")
    contents = await file.read()
    result = ingest_document(contents, file.filename)
    return {"message": f"Document indexed successfully.", "chunks": result["chunks"]}

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    chain = get_rag_chain()
    result = chain.invoke({"question": request.question})
    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"]
    )

@app.get("/documents")
def list_documents():
    from app.rag.vectorstore import get_vectorstore
    vs = get_vectorstore()
    return {"documents": vs.list_documents()}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
