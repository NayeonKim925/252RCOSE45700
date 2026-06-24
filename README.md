# LangChain RAG Chatbot

PDF 문서를 업로드하면 실시간으로 질문에 답변하는 RAG(Retrieval-Augmented Generation) 기반 챗봇입니다.

---

## 주요 기능

- **PDF 업로드** — 드래그&드롭 또는 클릭으로 PDF를 업로드하면 자동으로 벡터 인덱싱
- **실시간 스트리밍** — GPT-4o-mini의 응답을 토큰 단위로 스트리밍 출력 (타이핑 애니메이션)
- **출처 표시** — 답변 근거가 된 PDF 페이지 링크를 함께 제공
- **마크다운 렌더링** — 코드 블록, 표, 목록 등 서식이 있는 답변 지원
- **RAG / 일반 LLM 전환** — 사이드바 토글로 RAG 모드와 일반 GPT 모드 전환 가능

---

## 아키텍처

```
사용자 브라우저 (index.html)
        │  POST /upload          POST /ask-stream
        ▼                              ▼
  FastAPI 서버 (main.py)
        │                              │
   PDF 저장 + 인덱싱           벡터 검색 + LLM 호출
        │                              │
  PyPDFLoader                   Chroma DB (로컬)
  RecursiveTextSplitter               │
  OpenAI Embeddings            LangChain retriever (k=4)
        │                              │
    chroma_db/               GPT-4o-mini (streaming)
```

---

## 기술 스택

| 구분 | 기술 |
|------|------|
| 백엔드 | FastAPI, Python 3.11+ |
| LLM | OpenAI GPT-4o-mini |
| 임베딩 | OpenAI text-embedding-3-small |
| 벡터 DB | Chroma (로컬 퍼시스턴트) |
| RAG 프레임워크 | LangChain |
| PDF 파싱 | PyPDFLoader |
| 프론트엔드 | Vanilla JS, marked.js (마크다운 렌더링) |

---

## 설치 및 실행

### 1. 의존성 설치

```bash
pip install fastapi uvicorn langchain langchain-openai langchain-community \
            chromadb pypdf python-dotenv python-multipart
```

### 2. 환경 변수 설정

```bash
# .env 파일 생성
OPENAI_API_KEY=sk-...
```

### 3. (선택) 기존 PDF 일괄 인덱싱

```bash
# data/ 폴더에 PDF를 넣고 실행
python app/index.py
```

### 4. 서버 실행

```bash
uvicorn app.main:app --reload
```

브라우저에서 `http://localhost:8000` 접속 후 PDF를 업로드하고 질문하세요.

---

## API 엔드포인트

| Method | Path | 설명 |
|--------|------|------|
| GET | `/` | 챗봇 UI |
| GET | `/pdfs` | 업로드된 PDF 목록 |
| POST | `/upload` | PDF 업로드 및 자동 인덱싱 |
| POST | `/ask-stream` | RAG 스트리밍 답변 |
| GET | `/pdf/{filename}` | PDF 파일 서빙 |

---

## 프로젝트 구조

```
252RCOSE45700/
├── app/
│   ├── main.py       # FastAPI 서버 (업로드, RAG, 스트리밍)
│   └── index.py      # 일괄 PDF 인덱싱 스크립트
├── static/
│   └── index.html    # 챗봇 프론트엔드 (마크다운 지원)
├── data/             # 업로드된 PDF 저장 (git 미추적)
├── chroma_db/        # Chroma 벡터 DB (git 미추적)
└── .env              # API 키 (git 미추적)
```
