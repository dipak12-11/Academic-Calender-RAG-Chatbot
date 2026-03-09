from dotenv import load_dotenv
load_dotenv()
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import os


pc=Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name=os.getenv("PINECONE_INDEX_NAME")
index=pc.Index(name=index_name)
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embedding_model2= GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")#dim 3072

vector_store=PineconeVectorStore(embedding=embedding_model2,index=index)

query = "when is the SRIJAN?"
retriever_results = vector_store.similarity_search(query,k=3)


for res in retriever_results:
    print(f" {res.page_content} ")