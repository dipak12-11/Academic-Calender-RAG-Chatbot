import streamlit as st
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_pinecone import PineconeVectorStore
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage



st.set_page_config(page_title="Qwen3 RAG Chatbot", page_icon="🤖")
st.title("🤖 Qwen3 Agentic RAG Chatbot")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    st.info("Model: Qwen3-14B via Hugging Face")
    st.info("Embedding: all-MiniLM-L6-v2")
    st.info("Vector DB: Pinecone")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

@st.cache_resource(show_spinner="Initializing agent...")
def init_agent():
    """Initialize the RAG agent with Qwen3 model."""
    try:
        # 1. Initialize Pinecone
        pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
        index_name = st.secrets["PINECONE_INDEX_NAME"]
        
        if not index_name:
            st.error("PINECONE_INDEX_NAME not found in environment variables!")
            return None
            
        index = pc.Index(index_name)
        
        # 2. Initialize Embeddings
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        # 3. Initialize Vector Store
        vector_store = PineconeVectorStore(
            index=index,
            embedding=embedding_model,
        )
        
        # 4. Initialize Qwen3 LLM via Hugging Face Endpoint
        llm = HuggingFaceEndpoint(
            repo_id="Qwen/Qwen3-14B",  # ✅ Changed to Qwen3
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.1,
            huggingfacehub_api_token=st.secrets["HUGGINGFACEHUB_API_TOKEN"],
        )
        
        # 5. Wrap as Chat Model for LangGraph
        from langchain_huggingface import ChatHuggingFace
        model = ChatHuggingFace(llm=llm)
        
        # 6. Define RAG Tool with proper annotations
        @tool
        def retrieve_context(query: str):
            """Retrieve relevant documents from the Academic Calendar knowledge base.
            Use this tool when you need to find specific information from documents."""
            try:
                retrieved_docs = vector_store.similarity_search(query, k=5)
                if not retrieved_docs:
                    return "No relevant documents found in the knowledge base."
                
                serialized = "\n\n".join(
                    f"📄 Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content[:500]}"
                    for doc in retrieved_docs
                )
                return serialized
            except Exception as e:
                return f"Error retrieving documents: {str(e)}"
        
        # 7. Bind tools to model for proper tool calling
        model_with_tools = model.bind_tools([retrieve_context])
        
        # 8. Create ReAct Agent with system prompt
        system_prompt = """You are a helpful academic assistant with access to a knowledge base.
        
        INSTRUCTIONS:
        1. Always use the retrieve_context tool to search for information before answering
        2. Base your answers on the retrieved context when available
        3. If you cannot find information, be honest and say so
        4. Keep answers concise and well-structured
        5. Cite sources when referencing retrieved documents
        
        You have access to a tool that retrieves context from an Academic Calendar. 
        Use the tool to help answer user queries."""
        
        agent = create_react_agent(
            model=model_with_tools,  # ✅ Use model with bound tools
            tools=[retrieve_context],
            prompt=system_prompt,
        )
        
        return agent
        
    except Exception as e:
        st.error(f"Error initializing agent: {str(e)}")
        return None

# Initialize agent
agent = init_agent()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Handle user input
if question := st.chat_input("Ask a question about the Academic Calendar..."):
    if agent is None:
        st.error("Agent failed to initialize. Please check your environment variables.")
    else:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching knowledge base..."):
                try:
                    # Stream agent response
                    full_response = ""
                    tool_calls_shown = False
                    
                    for event in agent.stream(
                        {"messages": [HumanMessage(content=question)]},
                        stream_mode="values",
                    ):
                        # Check for messages in the event
                        if "messages" in event and event["messages"]:
                            last_message = event["messages"][-1]
                            
                            # Show tool calls (optional - for debugging)
                            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                                if not tool_calls_shown:
                                    with st.expander("🔧 Tool Calls"):
                                        for tc in last_message.tool_calls:
                                            st.json(tc)
                                    tool_calls_shown = True
                            
                            # Get AI response content
                            if hasattr(last_message, "content") and last_message.content:
                                if isinstance(last_message, AIMessage):
                                    full_response = last_message.content
                    
                    # Display final response
                    if full_response:
                        st.write(full_response)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": full_response
                        })
                    else:
                        st.write("I couldn't generate a response. Please try again.")
                        
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": f"Sorry, I encountered an error: {str(e)}"
                    })