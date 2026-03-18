from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from app.rag.vectorstore import get_vectorstore
from app.core.config import settings

PROMPT = """You are PrivateGPT Enterprise. Answer ONLY from the context below.
After every claim cite source like: [Source: filename.pdf]
If answer not in context say: "This information is not available in the uploaded documents."

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

def format_docs(docs):
    return "\n\n".join(
        f"[Source: {doc.metadata.get('source','unknown')}]\n{doc.page_content}"
        for doc in docs
    )

def get_rag_chain():
    vs = get_vectorstore()
    retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k": 5})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=settings.OPENAI_API_KEY)
    prompt = PromptTemplate(input_variables=["context", "question"], template=PROMPT)
    retrieved = []
    def retrieve(q):
        docs = retriever.invoke(q)
        retrieved.clear(); retrieved.extend(docs)
        return format_docs(docs)
    chain = ({"context": retrieve, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
    def invoke(inputs):
        answer = chain.invoke(inputs["question"])
        sources = list(set(d.metadata.get("source","unknown") for d in retrieved))
        return {"answer": answer, "sources": sources, "confidence": 0.9 if sources else 0.1}
    return type("R", (), {"invoke": staticmethod(invoke)})()
