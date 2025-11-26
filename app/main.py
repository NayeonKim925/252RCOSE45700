from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# 절대 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = str(BASE_DIR / "chroma_db")
STATIC_DIR = str(BASE_DIR / "static")

# FastAPI 앱
app = FastAPI(title="RAG Chatbot with LangChain & Chroma")

# static 폴더
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def read_root():
    return FileResponse(str(Path(STATIC_DIR) / "index.html"))


# 1. 벡터스토어 로드
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectordb = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
)
retriever = vectordb.as_retriever(search_kwargs={"k": 4})

# 2. LLM 설정
llm = ChatOpenAI(
    model="gpt-4o-mini",  
    temperature=0.2,
)


# 3. RAG 함수
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


# 4. 일반 질문 함수 (RAG 없이)
def simple_answer(question: str):
    response = llm.invoke(question)
    return response.content, []


# ----- 요청/응답 스키마 정의 -----
class Question(BaseModel):
    question: str
    use_rag: bool = True


class Answer(BaseModel):
    answer: str
    sources: list[dict]


# ----- /ask 엔드포인트 구현 -----
@app.post("/ask", response_model=Answer)
def ask(q: Question):
    if q.use_rag:
        answer, source_docs = rag_answer(q.question)
    else:
        answer, source_docs = simple_answer(q.question)

    sources = []
    for doc in source_docs:
        meta = doc.metadata
        sources.append(
            {
                "title": meta.get("source", "pdf"),
                "page": meta.get("page"),
                "url": meta.get(
                    "url",
                    f"/static/{meta.get('source', 'pdf')}#page={meta.get('page')}",
                ),
            }
        )

    return {
        "answer": answer,
        "sources": sources,
    }