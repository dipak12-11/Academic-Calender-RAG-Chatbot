import streamlit as st
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent

load_dotenv()

st.title("RAG Chatbot")

@st.cache_resource(show_spinner="Loading model...")
def init_agent():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = PineconeVectorStore(
        index=pc.Index(os.getenv("PINECONE_INDEX_NAME")),
        embedding=embedding_model,
    )
    llm = HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-R1-0528",
        task="text-generation",
        provider="auto",
    )
    model = ChatHuggingFace(llm=llm)

    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """Retrieve information to help answer a query."""
        retrieved_docs = vector_store.similarity_search(query, k=5)
        serialized = "\n\n".join(
            f"Source: {doc.metadata}\nContent: {doc.page_content}"
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    agent = create_react_agent(
        model,
        [retrieve_context],
        prompt="You have access to a tool that retrieves context from an Academic Calender. Use the tool to help answer user queries.",
    )
    return agent

agent = init_agent()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if question := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            final_answer = ""
            for event in agent.stream(
                {"messages": [{"role": "user", "content": question}]},
                stream_mode="values",
            ):
                last = event["messages"][-1]
                if hasattr(last, "type") and last.type == "ai":
                    final_answer = last.content

        st.write(final_answer)
        st.session_state.messages.append({"role": "assistant", "content": final_answer})