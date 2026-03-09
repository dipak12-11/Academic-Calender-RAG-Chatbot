import requests
import tempfile
from dotenv import load_dotenv
load_dotenv()
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter,Language
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings


url = "https://people.iitism.ac.in/~academics/assets/academic_files/AC%202025-26%20V12.pdf"
def load_pdf_from_url(url: str):
    """
    Docstring for load_pdf_from_url
    
    :param url: Description
    :type url: str
    """
    response = requests.get(url)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
        tmp_file.write(response.content)
        tmp_file_path = tmp_file.name
        
    loader = PyMuPDF4LLMLoader(tmp_file_path, mode='page', table_strategy="lines_strict")
    docs = loader.load()
    
    splitter=RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN,
        chunk_size=500,
        chunk_overlap=100
        )
    final_chunks = splitter.split_documents(docs)
    print(f"Loaded {len(docs)} documents from the PDF.")
    print(f"Split into {len(final_chunks)} chunks.")
    
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")# dim 384
    # embedding_model2= GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")#dim 3072
    pc=Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name=os.getenv("PINECONE_INDEX_NAME")
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384, 
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1" 
            )
        )
        print(f"Created Pinecone index: {index_name}")
    else:
        print(f"Pinecone index '{index_name}' already exists.")
    vector_store=PineconeVectorStore.from_documents(
        documents=final_chunks,
        embedding=embedding_model,
        index_name=index_name,
    )
    print(f"✓ Successfully stored {len(final_chunks)} chunks in Pinecone!")
    return vector_store
        
    
load_pdf_from_url(url)