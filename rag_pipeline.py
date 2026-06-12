import os
import sys
from dotenv import load_dotenv

# LangChain Imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables (like HPC_AI_API_KEY)
load_dotenv()

def build_rag_pipeline(file_path: str):
    """Builds and returns a RAG chain based on the provided text file."""
    
    print(f"--- Step 1: Loading Document '{file_path}' ---")
    if not os.path.exists(file_path):
        print(f"Error: The file {file_path} was not found.")
        sys.exit(1)
        
    loader = TextLoader(file_path)
    docs = loader.load()
    
    # We split the text into manageable chunks so the LLM context window isn't overwhelmed
    # and the semantic search is more precise.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    print(f"Successfully split the document into {len(splits)} chunks.\n")

    print("--- Step 2: Embedding & Vector Storage ---")
    # We use a free, lightweight, and robust HuggingFace model for embeddings.
    # We use this to convert our text chunks into vectors.
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Store these chunks in ChromaDB (an in-memory / local vector database)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    print("Vectors successfully embedded and stored in ChromaDB.\n")
    
    print("--- Step 3: Configuring Retrieval ---")
    # Setup the retriever to fetch the top 3 most relevant chunks based on semantic similarity
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    print("--- Step 4: Configuring Generation (HPC-AI LLM) ---")
    # Ensure HPC_AI_API_KEY is available
    if not os.environ.get("HPC_AI_API_KEY"):
        print("Error: HPC_AI_API_KEY environment variable is not set. Please set it in a .env file or export it in your terminal.")
        sys.exit(1)
        
    # Initialize the OpenAI-compatible Chat Model for HPC-AI
    # NOTE: You can change this to any model from your list (e.g., openai/gpt-5.5, anthropic/claude-opus-4.8)
    llm = ChatOpenAI(
        model="deepseek/deepseek-v4-flash", 
        openai_api_base="https://api.hpc-ai.com/inference/v1",
        openai_api_key=os.environ.get("HPC_AI_API_KEY")
    )
    
    # Define the system prompt for the LLM instructing it to use the retrieved context
    system_prompt = (
        "You are a helpful assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer based on the context, say that you don't know. "
        "Keep the answer clear and concise.\n\n"
        "Context: {context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # Create the RAG chain by combining the retrieval step and the QA generation step
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    print("RAG Pipeline is ready!\n")
    return rag_chain

def main():
    print("==================================================")
    print("   Basic RAG System with LangChain & HPC-AI API   ")
    print("==================================================\n")
    
    # We will use the sample.txt file created in the same directory
    file_name = "sample.txt"
    
    try:
        rag_chain = build_rag_pipeline(file_name)
    except Exception as e:
        print(f"Failed to build RAG pipeline: {e}")
        return

    print("Type 'exit' or 'quit' to stop.\n")
    
    while True:
        query = input("Ask a question about the document: ")
        if query.lower() in ["exit", "quit"]:
            print("Exiting...")
            break
            
        if not query.strip():
            continue
            
        print("\nThinking...")
        try:
            # Execution! Pass the query to the RAG chain
            response = rag_chain.invoke({"input": query})
            
            print("\n--- Answer ---")
            print(response["answer"])
            print("--------------\n")
            
        except Exception as e:
            print(f"An error occurred during generation: {e}")

if __name__ == "__main__":
    main()
