import requests
import tempfile
import hashlib
import os
from bs4 import BeautifulSoup
from urllib.parse import quote
from dotenv import load_dotenv
load_dotenv()
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec


def get_latest_calendar_url() -> str:
    base_url = "https://people.iitism.ac.in/~academics/"
    response = requests.get(base_url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for link in soup.find_all("a", href=True):
        if "Academic" in link.text and ".pdf" in link["href"]:
            pdf_url = link["href"]
            if not pdf_url.startswith("http"):
                pdf_url = base_url + pdf_url.lstrip("/")
            parts = pdf_url.split("iitism.ac.in/")
            pdf_url = f"https://people.iitism.ac.in/{quote(parts[1])}"
            return pdf_url
    raise ValueError("Could not find Academic Calendar PDF on the page!")


def get_pdf_hash(url: str) -> str:
    response = requests.get(url)
    return hashlib.md5(response.content).hexdigest()


def load_pdf_if_changed():
    url = get_latest_calendar_url()
    print(f"📄 Found calendar: {url}")

    new_hash = get_pdf_hash(url)
    hash_file = "last_pdf_hash.txt"

    if os.path.exists(hash_file):
        with open(hash_file, "r") as f:
            old_hash = f.read().strip()
        if old_hash == new_hash:
            print("✅ PDF unchanged — skipping re-index.")
            return

    print("🔄 PDF changed! Re-indexing...")

    response = requests.get(url)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
        tmp_file.write(response.content)
        tmp_file_path = tmp_file.name

    try:
        loader = PyMuPDF4LLMLoader(tmp_file_path, mode='page', table_strategy="lines_strict")
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.MARKDOWN,
            chunk_size=500,
            chunk_overlap=100
        )
        final_chunks = splitter.split_documents(docs)
        print(f"Split into {len(final_chunks)} chunks.")

        embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")# dim 384
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index_name = os.getenv("PINECONE_INDEX_NAME")

        if index_name in pc.list_indexes().names():
            print("🗑️ Clearing old Pinecone data...")
            try:
                pc.Index(index_name).delete(delete_all=True)
                pritn("✓ Old data cleared.")
            except Exception:
                pass
        else:
            pc.create_index(
                name=index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud=os.getenv("PINECONE_CLOUD", "aws"),
                                    region=os.getenv("PINECONE_REGION", "us-east-1"))
            )

        PineconeVectorStore.from_documents(
            documents=final_chunks,
            embedding=embedding_model,
            index_name=index_name,
        )
        print(f"✓ Stored {len(final_chunks)} chunks in Pinecone!")

        with open(hash_file, "w") as f:
            f.write(new_hash)

    finally:
        os.remove(tmp_file_path)  


if __name__ == "__main__":
    load_pdf_if_changed()
