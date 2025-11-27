import os
from pathlib import Path
from dotenv import load_dotenv
import time

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = str(BASE_DIR / "chroma_db")

PDF_FILES = [
    DATA_DIR / "컴퓨터활용능력 실기 핵심 요약노트.pdf",
    DATA_DIR / "컴퓨터활용능력 필기 핵심 요약노트.pdf",
]


def remove_chroma_db(db_path):
    import shutil
    import stat
    
    def handle_remove_readonly(func, path, exc):
        os.chmod(path, stat.S_IWRITE)
        func(path)
    
    if Path(db_path).exists():
        print(f"  Removing existing DB...")
        try:
            for attempt in range(3):
                try:
                    shutil.rmtree(db_path, onerror=handle_remove_readonly)
                    print(f"  DB removed successfully")
                    break
                except PermissionError as e:
                    if attempt < 2:
                        print(f"  Retry {attempt + 1}/3...")
                        time.sleep(1)
                    else:
                        print(f"  Warning: Could not remove DB")
                        break
        except Exception as e:
            print(f"  Warning: {e}")


def build_index():
    all_docs = []

    for pdf_path in PDF_FILES:
        pdf_path = Path(pdf_path)  

        print(f"Loading PDF: {pdf_path.name}")
        print(f"  Exists? {pdf_path.is_file()}")

        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()

        for i, doc in enumerate(pages):
            doc.metadata["source"] = pdf_path.name    
            if "page" not in doc.metadata:
                doc.metadata["page"] = i + 1
            doc.metadata["file_path"] = str(pdf_path)

        all_docs.extend(pages)

    print(f"Total pages loaded: {len(all_docs)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = splitter.split_documents(all_docs)

    print(f"Total chunks created: {len(chunks)}")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    print(f"Creating Chroma DB at: {CHROMA_DIR}")
    
    remove_chroma_db(CHROMA_DIR)
    
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    
    print(f"Chroma index built successfully with {len(chunks)} chunks")
    print(f"Index saved to: {CHROMA_DIR}")
    
    try:
        test_results = vectordb.similarity_search("컴퓨터", k=1)
        if test_results:
            print(f"Test search successful: {len(test_results)} results found")
        else:
            print("Warning: Test search returned no results")
    except Exception as e:
        print(f"Test search failed: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("Building Vector Index for RAG Chatbot")
    print("=" * 60)
    try:
        build_index()
        print("=" * 60)
        print("Index building completed!")
        print("=" * 60)
    except Exception as e:
        print("=" * 60)
        print(f"Error occurred: {e}")
        print("=" * 60)
        raise