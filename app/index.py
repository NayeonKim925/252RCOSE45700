# app/index.py

import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()  # OPENAI_API_KEY 읽기

# 1. 절대 경로로 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = str(BASE_DIR / "chroma_db")

# 2. 절대 경로에 PDF 파일 리스트 생성
PDF_FILES = [
    DATA_DIR / "컴퓨터활용능력 실기 핵심 요약노트.pdf",
    DATA_DIR / "컴퓨터활용능력 필기 핵심 요약노트.pdf",
]


def build_index():
    all_docs = []

    for pdf_path in PDF_FILES:
        pdf_path = Path(pdf_path)  

        # 디버깅용 출력
        print(f"Loading PDF: {pdf_path}")
        print(f"  Exists? {pdf_path.is_file()}")

        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")

        # 1. PDF 로드
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()  # 페이지 단위의 Document 리스트

        # 2. 각 페이지에서 출처 정보 달기
        for i, doc in enumerate(pages):
            doc.metadata.setdefault("source", os.path.basename(str(pdf_path)))
            doc.metadata["page"] = doc.metadata.get("page", i + 1)

        all_docs.extend(pages)

    print(f"Total pages loaded: {len(all_docs)}")

    # 3. 텍스트 chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(all_docs)

    print(f"Total chunks created: {len(chunks)}")

    # 4. OpenAI 임베딩
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 5. Chroma 벡터스토어 생성
    print(f"Creating Chroma DB at: {CHROMA_DIR}")
    vectordb = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    vectordb.persist()
    print(f"Chroma index built successfully with {len(chunks)} chunks.")


if __name__ == "__main__":
    build_index()