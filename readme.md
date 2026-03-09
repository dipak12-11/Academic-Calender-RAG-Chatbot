# 🎓 Academic Calendar RAG Chatbot
### An intelligent, self-updating chatbot for IIT ISM Dhanbad's Academic Calendar — powered by Qwen3, Pinecone, and LangChain.

---

## 🧠 What Is This?

A **Retrieval-Augmented Generation (RAG)** chatbot that lets students and faculty ask natural language questions about the IIT ISM Dhanbad academic calendar — and always get answers from the **latest version** of the document.

No more manually hunting through PDFs. Just ask.

> *"When does the summer semester end?"*

---

## ✨ Features

| Feature | Details |
|---|---|
| 🤖 **LLM** | Qwen3-14B via Hugging Face Inference API |
| 🔍 **Retrieval** | Pinecone vector store with semantic search |
| 📄 **Document Loader** | PyMuPDF4LLM with Markdown-aware chunking |
| 🧬 **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` |
| 🔄 **Auto-refresh** | GitHub Actions scrapes & re-indexes daily |
| 💬 **Multi-turn chat** | Sliding window conversation history |
| 🌐 **UI** | Streamlit web interface |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    GitHub Actions                        │
│   (runs daily at 6AM UTC)                               │
│                                                         │
│   Scrape IIT ISM website                                │
│        ↓                                                │
│   Hash check — PDF changed?                             │
│        ↓ (yes)                                          │
│   Download PDF → Chunk → Embed → Upsert to Pinecone     │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                   Streamlit App                          │
│                                                         │
│   User asks question                                    │
│        ↓                                                │
│   LangGraph ReAct Agent                                 │
│        ↓                                                │
│   retrieve_context tool → Pinecone similarity search    │
│        ↓                                                │
│   Qwen3-14B generates answer from retrieved chunks      │
│        ↓                                                │
│   Response displayed in chat UI                         │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
Academic-Calendar-RAG-Chatbot/
│
├── A.Loader.py              # PDF scraper, chunker, and Pinecone indexer
├── B.Qwen-3-Chatbot.py      # Streamlit chatbot UI + LangGraph agent
│
├── .github/
│   └── workflows/
│       └── refresh.yml      # GitHub Actions daily auto-refresh
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/dipak12-11/Academic-Calender-RAG-Chatbot.git
cd Academic-Calender-RAG-Chatbot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables
Create a `.env` file in the root directory:
```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_index_name
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
HUGGINGFACEHUB_API_TOKEN=your_hf_token
```

### 4. Run the loader (first time only)
```bash
python A.Loader.py
```
This will scrape the IIT ISM website, download the latest academic calendar PDF, chunk it, embed it, and push everything to Pinecone.

### 5. Launch the chatbot
```bash
streamlit run B.Qwen-3-Chatbot.py
```

---

## 🔄 Auto-Refresh (GitHub Actions)

The loader runs automatically every day via GitHub Actions. It:

1. Scrapes the IIT ISM academics page for the latest PDF URL
2. Computes an MD5 hash of the PDF
3. Compares it to the previously stored hash
4. **Skips** re-indexing if nothing changed (saves API quota)
5. **Wipes and re-indexes** Pinecone if the PDF is updated

To set this up on your fork, add these repository secrets under **Settings → Secrets → Actions**:

- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME`
- `HUGGINGFACEHUB_API_TOKEN`

You can also trigger a manual refresh anytime from the **Actions** tab.

---

## 🔧 How the RAG Pipeline Works

```
PDF (online)
    │
    ▼
PyMuPDF4LLM             → Extracts text preserving tables & layout
    │
    ▼
RecursiveCharacterTextSplitter  → Chunks by Markdown structure
(chunk_size=500, overlap=100)     keeping semantic units intact
    │
    ▼
sentence-transformers           → Converts chunks to 384-dim vectors
(all-MiniLM-L6-v2)
    │
    ▼
Pinecone VectorStore            → Stores and indexes all vectors
    │
    ▼
LangGraph ReAct Agent           → On user query:
    │                              1. Calls retrieve_context tool
    │                              2. Gets top-5 similar chunks
    │                              3. Feeds to Qwen3-14B
    ▼
Answer grounded in document
```

---

## 💬 Example Queries

```
"What is the Convocation?"
"When does the mid-semester break start?"
"What is the last date to submit fees in winter Sem?"
"When is SRIJAN?"
"What is the last date to withdraw from a course?"
```

---

## 🛡️ Key Design Decisions

**Why delete + re-index instead of updating?**
The academic calendar is a full document — if a date changes anywhere, partial updates risk leaving stale chunks alongside new ones. A full wipe guarantees consistency.

**Why hash checking?**
Embedding + upserting 100+ chunks consumes Pinecone and HuggingFace API quota. On days the PDF hasn't changed (most days), the job exits in ~5 seconds.

**Why sliding window instead of full history?**
Passing the full conversation history to the LLM causes hallucinations after 3–4 turns — earlier answers start influencing current ones. A window of the last 4 messages keeps context without drift.

---

## 📦 Requirements

```
requests
beautifulsoup4
python-dotenv
langchain-pymupdf4llm
langchain-text-splitters
langchain-huggingface
langchain-pinecone
langchain-google-genai
pinecone
sentence-transformers
streamlit
langgraph
```

---

## 🙏 Acknowledgements

- [IIT ISM Dhanbad](https://www.iitism.ac.in/) for the academic calendar
- [Qwen3](https://huggingface.co/Qwen/Qwen3-14B) by Alibaba Cloud
- [Pinecone](https://www.pinecone.io/) for vector storage
- [LangChain](https://www.langchain.com/) & [LangGraph](https://langchain-ai.github.io/langgraph/) for the agent framework

---

<p align="center">Built with ❤️ by <a href="https://github.com/dipak12-11">dipak12-11</a></p>