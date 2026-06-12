# 🔍 RAG Foundation

A full-stack **Retrieval-Augmented Generation (RAG)** application that lets you upload documents and chat with them using a powerful LLM — with automatic internet fallback when the answer isn't in your documents.

---

## ✨ Features

- 📄 **Document Upload** — Supports `.txt` and `.pdf` files
- 🧠 **Semantic Search** — Embeds documents using `all-MiniLM-L6-v2` (HuggingFace) and stores vectors in ChromaDB
- 💬 **Conversational Chat** — Multi-turn chat with full history context
- 🌐 **Web Search Fallback** — If the answer isn't in your documents, the assistant offers to search the internet via DuckDuckGo
- ⚡ **HPC-AI LLM Backend** — Uses `deepseek/deepseek-v4-flash` via an OpenAI-compatible API
- 🖥️ **React Frontend** — Clean UI built with React + Vite

---

## 🏗️ Architecture

```
rag_foundation/
├── rag_pipeline.py       # Standalone CLI RAG pipeline
├── sample.txt            # Sample document for testing
├── requirements.txt      # Python dependencies
├── backend/
│   └── main.py           # FastAPI server (upload + chat endpoints)
└── frontend/
    ├── src/
    │   ├── App.jsx       # Main React app
    │   └── App.css       # Styles
    └── package.json      # Node dependencies
```

### How It Works

```
User Query
    │
    ▼
[Frontend] ──── POST /chat ────► [FastAPI Backend]
                                        │
                              ┌─────────┴──────────┐
                              │                    │
                     [ChromaDB Retriever]    [Chat History]
                              │
                    Top-3 Relevant Chunks
                              │
                              ▼
                    [DeepSeek LLM via HPC-AI]
                              │
                    ┌─────────┴──────────┐
               Answer Found?        Not Found?
                    │                    │
                Return Answer    Ask to Search Web
                                         │
                                  [DuckDuckGo Search]
                                         │
                                  Return Web Answer
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- An **HPC-AI API Key** from [hpc-ai.com](https://hpc-ai.com)

---

### 1. Clone the Repository

```bash
git clone https://github.com/nihithkothakota/RAG.git
cd RAG
```

### 2. Set Up Python Backend

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the root directory:

```env
HPC_AI_API_KEY=your_hpc_ai_api_key_here
```

### 4. Run the Backend

```bash
uvicorn backend.main:app --reload
```

The API will be available at `http://localhost:8000`.

### 5. Run the Frontend

```bash
cd frontend
npm install
npm run dev
```

The UI will be available at `http://localhost:5173`.

---

## 🖥️ CLI Mode (Standalone Pipeline)

You can also run the RAG pipeline directly from the terminal without the UI:

```bash
python rag_pipeline.py
```

This loads `sample.txt`, builds the vector store, and starts an interactive Q&A loop in your terminal.

---

## 📡 API Endpoints

| Method | Endpoint  | Description                        |
|--------|-----------|------------------------------------|
| `POST` | `/upload` | Upload a `.txt` or `.pdf` document |
| `POST` | `/chat`   | Send a query with chat history     |

### `/upload` — Request
```
multipart/form-data
  file: <your .txt or .pdf file>
```

### `/chat` — Request Body
```json
{
  "query": "What is the main topic of the document?",
  "history": [
    { "type": "user", "text": "Hello!" },
    { "type": "bot",  "text": "Hi! Upload a document to get started." }
  ]
}
```

---

## 🛠️ Tech Stack

| Layer      | Technology                                      |
|------------|-------------------------------------------------|
| LLM        | DeepSeek V4 Flash via HPC-AI (OpenAI-compatible)|
| Embeddings | `all-MiniLM-L6-v2` (HuggingFace)               |
| Vector DB  | ChromaDB                                        |
| Framework  | LangChain                                       |
| Backend    | FastAPI + Uvicorn                               |
| Frontend   | React + Vite                                    |
| Web Search | DuckDuckGo (via `langchain-community`)          |

---

## 📦 Dependencies

```
langchain>=0.2.0
langchain-openai
langchain-community
langchain-huggingface
langchain-chroma
chromadb
python-dotenv
sentence-transformers
fastapi
uvicorn
python-multipart
pypdf
ddgs
```

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).
