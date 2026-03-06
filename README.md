# ⚡ Local Multimodal Agentic RAG (AxIn Help)

A secure, fully local, multi-agent AI pipeline designed to query proprietary enterprise documentation with zero data leakage. Built with **LangGraph**, **Ollama**, and **Streamlit**, this system utilizes an advanced **Model Tiering (LLM Cascading)** architecture to balance lightning-fast user interactions with highly accurate, hallucination-free document retrieval.

## 🧠 Core Architecture: Model Tiering
Running LLMs locally on constrained hardware (e.g., 16GB RAM) requires strategic resource management. Instead of bottlenecking the system with a single massive model, this pipeline dynamically routes queries to specialized local models:

1. **The Traffic Cop (`llama3.2:1b`)**: An ultra-fast, lightweight model dedicated strictly to intent classification. It instantly routes user prompts to the correct processing node (RAG, WEB, or CHAT).
2. **The Conversationalist (`llama3.2:1b`)**: Handles basic greetings, small talk, and live web summaries with near-zero latency.
3. **The Subject Matter Expert (`llama3.1 8B`)**: The heavy lifter. Reserved strictly for the RAG route, this model reads retrieved proprietary context and generates highly accurate, perfectly formatted (markdown/bulleted) technical instructions.

## ✨ Key Features
* **Agentic Routing:** Automatically categorizes user queries to prevent unnecessary database searches or hallucinations.
* **Semantic Search Boosting:** Enhances base retrieval by dynamically augmenting queries with action-oriented keywords (e.g., *how to use, navigate, access, steps*) to bridge the gap between user vocabulary and technical manual terminology.
* **100% Local & Secure:** No API keys, no cloud processing. All embeddings and generations happen on bare metal, ensuring proprietary documents remain strictly confidential.
* **Real-Time Streaming UI:** A custom-styled Streamlit interface that mimics high-end SaaS platforms. It features dynamic state-tracking pills, right-aligned user chat bubbles, and real-time token streaming to mask background processing time.

## 🛠️ Tech Stack
* **Orchestration:** LangGraph, LangChain
* **Local LLM Execution:** Ollama
* **Vector Database:** ChromaDB
* **Embeddings:** `nomic-embed-text`
* **Web Search:** DuckDuckGo Search API
* **Frontend:** Streamlit

## 🚀 Quick Start Guide

### 1. Prerequisites
Ensure you have Python 3.9+ and [Ollama](https://ollama.com/) installed on your machine.

### 2. Pull Required Local Models
Open your terminal and pull the necessary models into your Ollama instance:
```bash
ollama pull llama3.2:1b
ollama pull llama3.1
ollama pull nomic-embed-text
```
### 3. Installation
Clone the repository and install the required dependencies:
```bash
git clone [https://github.com/R-Saad007/Local-Multimodal-Agentic-RAG.git](https://github.com/R-Saad007/Local-Multimodal-Agentic-RAG.git)
cd Local-Multimodal-Agentic-RAG
python -m venv env
# On Windows use: .\env\Scripts\activate
source env/bin/activate  
pip install -r requirements.txt
```
### 4. Database Setup
Note: Proprietary PDFs and the compiled <code>chroma_db</code> folder are intentionally excluded from this repository for security purposes.
To test the RAG capabilities, run your own ingestion script to populate a local <code>chroma_db</code> directory with your specific markdown or PDF documentation using the <code>nomic-embed-text</code> embedding model.

### 5. Launch the Application
For a seamless startup, a batch script is included to wake up the Ollama server, activate the environment, and launch the Streamlit app automatically:
```bash
# On Windows
.\launch_demo.bat

# Or run manually:
streamlit run app.py
```
