from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from urllib.parse import quote
import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = str(BASE_DIR / "chroma_db")
STATIC_DIR = str(BASE_DIR / "static")
DATA_DIR = BASE_DIR / "data"

app = FastAPI(title="RAG Chatbot with LangChain & Chroma")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def read_root():
    return FileResponse(str(Path(STATIC_DIR) / "index.html"))


@app.get("/pdf/{filename:path}")
def serve_pdf(filename: str):
    pdf_path = (DATA_DIR / filename).resolve()
    
    if not str(pdf_path).startswith(str(DATA_DIR.resolve())):
        raise HTTPException(status_code=403, detail="Access denied")
    
    if not pdf_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    
    if not pdf_path.suffix.lower() == '.pdf':
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=pdf_path.name
    )


embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectordb = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
)
retriever = vectordb.as_retriever(search_kwargs={"k": 4})

llm = ChatOpenAI(
    model="gpt-4o-mini",  
    temperature=0.2,
)


def rag_answer(question: str):
    docs = retriever.invoke(question)

    context_parts = []
    for d in docs:
        context_parts.append(d.page_content)
    context = "\n\n---\n\n".join(context_parts)

    prompt = (
        "다음 컨텍스트를 바탕으로 사용자의 질문에 답변해 주세요. "
        "모르면 모른다고 말해도 됩니다.\n\n"
        f"컨텍스트:\n{context}\n\n"
        f"질문: {question}\n\n"
        "답변:"
    )

    response = llm.invoke(prompt)
    answer_text = response.content

    return answer_text, docs


def simple_answer(question: str):
    response = llm.invoke(question)
    return response.content, []


class Question(BaseModel):
    question: str
    use_rag: bool = True


class Answer(BaseModel):
    answer: str
    sources: list[dict]


@app.post("/ask", response_model=Answer)
def ask(q: Question):
    if q.use_rag:
        answer, source_docs = rag_answer(q.question)
    else:
        answer, source_docs = simple_answer(q.question)

    seen_sources = set()
    sources = []
    
    for doc in source_docs:
        meta = doc.metadata
        source_file = meta.get("source", "unknown.pdf")
        page_num = meta.get("page", 1)
        
        source_key = (source_file, page_num)
        
        if source_key not in seen_sources:
            seen_sources.add(source_key)
            
            display_name = source_file.replace('.pdf', '')
            encoded_filename = quote(source_file)
            
            sources.append({
                "title": display_name,  
                "page": page_num,
                "url": f"/pdf/{encoded_filename}#page={page_num}",
            })

    return {
        "answer": answer,
        "sources": sources,
    }