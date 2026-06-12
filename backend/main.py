import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from dotenv import load_dotenv

# LangChain Imports
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.tools import DuckDuckGoSearchRun

load_dotenv()

app = FastAPI()

# Allow frontend to access API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize global components
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")

# Setup generation
if not os.environ.get("HPC_AI_API_KEY"):
    print("Warning: HPC_AI_API_KEY environment variable is not set.")

llm = ChatOpenAI(
    model="deepseek/deepseek-v4-flash", 
    openai_api_base="https://api.hpc-ai.com/inference/v1",
    openai_api_key=os.environ.get("HPC_AI_API_KEY")
)

class ChatRequest(BaseModel):
    query: str
    history: List[Dict[str, str]] = []

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(('.txt', '.pdf')):
        raise HTTPException(status_code=400, detail="Only .txt and .pdf files are supported.")
    
    # Save the file temporarily
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename)
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Load the document
        if file.filename.endswith('.pdf'):
            loader = PyPDFLoader(temp_path)
        else:
            loader = TextLoader(temp_path)
            
        docs = loader.load()
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(docs)
        
        # Add to vector store
        vectorstore.add_documents(documents=splits)
        return {"message": f"Successfully uploaded and processed {file.filename}. Added {len(splits)} chunks."}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(temp_dir)

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        chat_history = []
        for msg in request.history:
            if msg["type"] == "user":
                chat_history.append(HumanMessage(content=msg["text"]))
            else:
                chat_history.append(AIMessage(content=msg["text"]))

        # Check if the last bot message asked for internet search
        asked_for_search = False
        if len(request.history) >= 1:
            last_bot = request.history[-1]
            if last_bot["type"] == "bot" and "search the internet" in last_bot["text"].lower():
                asked_for_search = True

        # If the user says "yes" to a web search prompt
        user_intent_yes = request.query.lower().strip() in ["yes", "yeah", "yep", "sure", "ok", "do it", "please", "search"]
        if asked_for_search and user_intent_yes:
            # Use the LLM to determine the actual factual query from history
            history_str = "\n".join([f"{msg['type']}: {msg['text']}" for msg in request.history[-5:]])
            
            query_extractor = ChatPromptTemplate.from_messages([
                ("system", "Analyze this conversation history and extract the core factual question the user is trying to find the answer to. Return ONLY the search query string, nothing else. If the user is just asking a meta-question like 'can you search the internet?', return 'general knowledge'."),
                ("human", "{history}")
            ])
            extractor_chain = query_extractor | llm
            actual_query = extractor_chain.invoke({"history": history_str}).content.strip()
            
            # Perform Web Search
            search = DuckDuckGoSearchRun()
            web_results = search.run(actual_query)
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a highly intelligent and helpful AI assistant.\n\n"
                           "You just searched the internet for the user's question because it wasn't in their local documents.\n"
                           "Provide a detailed, clear, and understandable explanation using the following internet search results:\n\n"
                           "Internet Context:\n{context}\n\n"
                           "Use markdown formatting to make your explanation highly readable."),
                ("human", "{input}")
            ])
            chain = prompt | llm
            response = chain.invoke({"context": web_results, "input": actual_query})
            return {"answer": response.content}

        # Normal RAG flow
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(request.query)
        context = "\n\n".join([d.page_content for d in docs]) if docs else "No relevant documents found."
        
        system_prompt = """You are a highly intelligent and helpful AI assistant.
Your task is to answer the user's question based ONLY on the provided context.
Provide your answer in a clear, highly understandable manner with a detailed explanation. Use markdown formatting (like bolding and bullet points) to make it easy to read.

CRITICAL RULES:
1. If the user asks a general conversational question (e.g., "hello", "can you search the internet?", "what can you do?"), you may answer naturally without needing context.
2. If the user asks a factual question and the answer is NOT found in the provided context, DO NOT guess or hallucinate. Instead, you MUST reply EXACTLY with: "I couldn't find the answer in the uploaded documents. Would you like me to search the internet?"

Context:
{context}"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        
        chain = prompt | llm
        response = chain.invoke({"context": context, "chat_history": chat_history, "input": request.query})
        
        return {"answer": response.content}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
