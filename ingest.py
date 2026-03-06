import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

def load_documents(directory_path="./docs"):
    """Loads all PDF documents from the specified directory."""
    print(f"Loading PDFs from {directory_path}...")
    loader = DirectoryLoader(directory_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} document pages.")
    return documents

def chunk_documents(documents):
    print("Chunking documents...")
    # Increased size to 1000 for better context retention
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def build_vector_store(chunks, embeddings_model, persist_directory="./chroma_db"):
    """Embeds chunks and stores them in a local Chroma database."""
    print("Embedding chunks and saving to ChromaDB. This might take a moment...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory=persist_directory
    )
    print(f"Successfully saved vectors to {persist_directory}")
    return vector_store

if __name__ == "__main__":
    # Initialize the local Ollama embedding model
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434"
    )
    
    # Execute the pipeline
    docs = load_documents()
    if docs:
        chunks = chunk_documents(docs)
        # Passing the embeddings model explicitly now
        build_vector_store(chunks, embeddings) 
    else:
        print("No documents found. Please add PDFs to the './docs' folder.")