from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel
from urllib.parse import quote
import json
import asyncio

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
MAX_HISTORY_TURNS = 8  # 유지할 최대 대화 턴 수

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


# ── 벡터 DB / LLM ──────────────────────────────────────────────────────────
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectordb = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
)
retriever = vectordb.as_retriever(search_kwargs={"k": 4})

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
llm_streaming = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, streaming=True)


# ── 대화 히스토리 ────────────────────────────────────────────────────────────
# 단일 세션 인메모리 히스토리: [{"role": "user"|"assistant", "content": "..."}]
conversation_history: list[dict] = []


def _format_history() -> str:
    """최근 MAX_HISTORY_TURNS 턴의 대화를 프롬프트용 문자열로 변환한다."""
    recent = conversation_history[-(MAX_HISTORY_TURNS * 2):]
    if not recent:
        return ""
    lines = []
    for msg in recent:
        role = "사용자" if msg["role"] == "user" else "AI"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def _append_history(question: str, answer: str):
    conversation_history.append({"role": "user", "content": question})
    conversation_history.append({"role": "assistant", "content": answer})


@app.delete("/history")
def reset_history():
    conversation_history.clear()
    return {"status": "cleared"}


# ── PDF 인덱싱 ────────────────────────────────────────────────────────────
def _index_pdf(pdf_path: Path):
    global vectordb, retriever

    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    for i, doc in enumerate(pages):
        doc.metadata["source"] = pdf_path.name
        if "page" not in doc.metadata:
            doc.metadata["page"] = i + 1

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
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


# ── 공통 유틸 ────────────────────────────────────────────────────────────
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


def _build_rag_prompt(question: str, context: str) -> str:
    history = _format_history()
    history_block = f"\n이전 대화:\n{history}\n" if history else ""
    return (
        "당신은 친절하고 정확한 AI 어시스턴트입니다. "
        "아래 참고 문서를 바탕으로 답변하세요. 모르면 모른다고 말해도 됩니다."
        f"{history_block}\n"
        f"참고 문서:\n{context}\n\n"
        f"사용자: {question}\nAI:"
    )


def _build_simple_prompt(question: str) -> str:
    history = _format_history()
    history_block = f"\n이전 대화:\n{history}\n" if history else ""
    return (
        "당신은 친절하고 정확한 AI 어시스턴트입니다."
        f"{history_block}\n"
        f"사용자: {question}\nAI:"
    )


# ── 답변 함수 ────────────────────────────────────────────────────────────
def rag_answer(question: str):
    docs = retriever.invoke(question)
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    response = llm.invoke(_build_rag_prompt(question, context))
    _append_history(question, response.content)
    return response.content, docs


async def rag_answer_stream(question: str):
    docs = retriever.invoke(question)
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    prompt = _build_rag_prompt(question, context)

    yield json.dumps({"type": "sources", "data": _build_sources(docs)}) + "\n"

    full_response = ""
    async for chunk in llm_streaming.astream(prompt):
        if chunk.content:
            full_response += chunk.content
            yield json.dumps({"type": "token", "data": chunk.content}) + "\n"
            await asyncio.sleep(0.01)

    _append_history(question, full_response)
    yield json.dumps({"type": "done"}) + "\n"


def simple_answer(question: str):
    response = llm.invoke(_build_simple_prompt(question))
    _append_history(question, response.content)
    return response.content, []


async def simple_stream(question: str):
    prompt = _build_simple_prompt(question)

    yield json.dumps({"type": "sources", "data": []}) + "\n"

    full_response = ""
    async for chunk in llm_streaming.astream(prompt):
        if chunk.content:
            full_response += chunk.content
            yield json.dumps({"type": "token", "data": chunk.content}) + "\n"
            await asyncio.sleep(0.01)

    _append_history(question, full_response)
    yield json.dumps({"type": "done"}) + "\n"


# ── 엔드포인트 ────────────────────────────────────────────────────────────
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
