from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel
from urllib.parse import quote
import os
import json
import asyncio
import shutil

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = str(BASE_DIR / "chroma_db")
STATIC_DIR = str(BASE_DIR / "static")
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

MAX_UPLOAD_MB = 50

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

    if pdf_path.suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Invalid file type")

    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=pdf_path.name,
    )


@app.get("/pdfs")
def list_pdfs():
    files = [f.name for f in DATA_DIR.glob("*.pdf")]
    return {"files": files}


embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectordb = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
)
retriever = vectordb.as_retriever(search_kwargs={"k": 4})

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
llm_streaming = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, streaming=True)


def _index_pdf(pdf_path: Path):
    """PDF를 읽어 기존 벡터 DB에 추가한다."""
    global vectordb, retriever

    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    for i, doc in enumerate(pages):
        doc.metadata["source"] = pdf_path.name
        if "page" not in doc.metadata:
            doc.metadata["page"] = i + 1

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    chunks = splitter.split_documents(pages)

    vectordb.add_documents(chunks)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    return len(chunks)


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드할 수 있습니다.")

    dest = DATA_DIR / file.filename
    size = 0
    with dest.open("wb") as f:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_MB * 1024 * 1024:
                dest.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"파일 크기가 {MAX_UPLOAD_MB}MB를 초과합니다.",
                )
            f.write(chunk)

    try:
        num_chunks = _index_pdf(dest)
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"인덱싱 중 오류: {e}")

    return JSONResponse({"filename": file.filename, "chunks": num_chunks})


def _build_sources(docs: list) -> list[dict]:
    seen = set()
    sources = []
    for doc in docs:
        meta = doc.metadata
        source_file = meta.get("source", "unknown.pdf")
        page_num = meta.get("page", 1)
        key = (source_file, page_num)
        if key not in seen:
            seen.add(key)
            sources.append(
                {
                    "title": source_file.replace(".pdf", ""),
                    "page": page_num,
                    "url": f"/pdf/{quote(source_file)}#page={page_num}",
                }
            )
    return sources


def rag_answer(question: str):
    docs = retriever.invoke(question)
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    prompt = (
        "다음 컨텍스트를 바탕으로 사용자의 질문에 답변해 주세요. "
        "모르면 모른다고 말해도 됩니다.\n\n"
        f"컨텍스트:\n{context}\n\n"
        f"질문: {question}\n\n"
        "답변:"
    )
    response = llm.invoke(prompt)
    return response.content, docs


async def rag_answer_stream(question: str):
    docs = retriever.invoke(question)
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    prompt = (
        "다음 컨텍스트를 바탕으로 사용자의 질문에 답변해 주세요. "
        "모르면 모른다고 말해도 됩니다.\n\n"
        f"컨텍스트:\n{context}\n\n"
        f"질문: {question}\n\n"
        "답변:"
    )

    yield json.dumps({"type": "sources", "data": _build_sources(docs)}) + "\n"

    async for chunk in llm_streaming.astream(prompt):
        if chunk.content:
            yield json.dumps({"type": "token", "data": chunk.content}) + "\n"
            await asyncio.sleep(0.01)

    yield json.dumps({"type": "done"}) + "\n"


def simple_answer(question: str):
    response = llm.invoke(question)
    return response.content, []


async def simple_stream(question: str):
    yield json.dumps({"type": "sources", "data": []}) + "\n"
    async for chunk in llm_streaming.astream(question):
        if chunk.content:
            yield json.dumps({"type": "token", "data": chunk.content}) + "\n"
            await asyncio.sleep(0.01)
    yield json.dumps({"type": "done"}) + "\n"


class Question(BaseModel):
    question: str
    use_rag: bool = True


class Answer(BaseModel):
    answer: str
    sources: list[dict]


@app.post("/ask", response_model=Answer)
def ask(q: Question):
    if q.use_rag:
        answer, docs = rag_answer(q.question)
    else:
        answer, docs = simple_answer(q.question)
    return {"answer": answer, "sources": _build_sources(docs)}


@app.post("/ask-stream")
async def ask_stream(q: Question):
    generator = rag_answer_stream(q.question) if q.use_rag else simple_stream(q.question)
    return StreamingResponse(generator, media_type="text/plain")
