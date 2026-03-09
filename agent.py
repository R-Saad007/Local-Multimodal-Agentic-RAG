from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.tools import DuckDuckGoSearchRun

# --- 1. INITIALIZE LOCAL LLMS & TOOLS ---

# 1. The Traffic Cop: Extremely fast routing
router_llm = ChatOllama(
    model="llama3.2:1b", 
    base_url="http://localhost:11434", 
    temperature=0, 
    num_ctx=2048
)

# 2. The Conversationalist: Fast generation for basic Chat and Web
fast_generator_llm = ChatOllama(
    model="llama3.2:1b", 
    base_url="http://localhost:11434", 
    temperature=0, 
    num_ctx=2048
)

# 3. The SME (Subject Matter Expert): Highly accurate 8B model strictly for RAG
smart_generator_llm = ChatOllama(
    model="llama3.1", 
    base_url="http://localhost:11434", 
    temperature=0, 
    num_ctx=4096
)

web_search = DuckDuckGoSearchRun()

embeddings = OllamaEmbeddings(
    model="nomic-embed-text", 
    base_url="http://localhost:11434"
)
vector_store = Chroma(
    persist_directory="./chroma_db", 
    embedding_function=embeddings
)

# --- 2. DEFINE THE STATE ---

class AgentState(TypedDict):
    query: str
    route: str
    context: str
    response: str

# --- 3. NODE FUNCTIONS ---

def router_node(state: AgentState):
    print("\n--- ROUTING QUERY ---")
    query = state["query"]
    query_lower = query.lower()
    
    # --- 1. THE GUARANTEED FAST-TRACK ---
    internal_keywords = ["axin", "clickops", "reporting hub", "e-noc", "enoc", "site watch", "site specific view"]
    if any(keyword in query_lower for keyword in internal_keywords):
        print("Decision: Fast-tracked to RAG (Internal Module Detected)")
        return {"route": "RAG"}
    
    # --- 2. THE PROBABILISTIC LLM ROUTER ---
    prompt = f"""Task: Route the user query to exactly ONE destination.

    DESTINATIONS:
    WEB: Weather, news, real-time data, current events, locations, cities, or countries.
    CHAT: Greetings (Hi, Hello), thanks, or small talk.
    RAG: Technical questions, troubleshooting, or internal manuals.
    
    Examples:
    "Hi" -> CHAT
    "What is the temperature in Lahore?" -> WEB
    "Who won the game?" -> WEB
    "How do I reset my password?" -> RAG
    "What is AxIn?" -> RAG
    "Thank you!" -> CHAT
    "How do I use the reporting hub?" -> RAG
    "What does clickops do?" -> RAG
    
    User Query: "{query}"
    Output strictly ONE word (WEB, CHAT, or RAG):"""
    
    decision = router_llm.invoke(prompt).content.upper()
    
    if "WEB" in decision:
        route = "WEB"
    elif "CHAT" in decision:
        route = "CHAT"
    else:
        route = "RAG" # Default fallback
        
    print(f"Decision string from LLM: {decision}")
    print(f"Final Route: {route}")
    return {"route": route}

def retrieve_rag_node(state: AgentState):
    print("--- RETRIEVING FROM LOCAL DATABASE ---")
    query = state["query"]
    query_lower = query.lower()
    
    # Massive Wide Net
    docs = vector_store.similarity_search(query, k=15)
    
    modules = ["reporting hub", "clickops", "site watch", "e-noc", "site specific view"]
    target_module = next((m for m in modules if m in query_lower), None)
    
    if target_module:
        print(f"Boosting search for module: {target_module}")
        
        # Boost 1: Action-oriented
        action_query = f"{target_module} {query} how to use navigate access steps login"
        action_docs = vector_store.similarity_search(action_query, k=10)
        
        # Boost 2: Data-oriented
        data_query = f"{target_module} {query} metrics KPI parameters category performance evaluates"
        data_docs = vector_store.similarity_search(data_query, k=10)
        
        docs = action_docs + data_docs + docs 

    # THE FIX: Deduplicate using the FULL content to bypass boilerplate headers
    unique_contents = []
    seen = set()
    for d in docs:
        if d.page_content not in seen:
            unique_contents.append(d.page_content)
            seen.add(d.page_content)

    context = "\n\n".join(unique_contents)
    print(f"--- ASSEMBLED CONTEXT LENGTH: {len(context)} characters ---")
    
    # THE TRUTH SERUM: Debug dump
    with open("debug_context.txt", "w", encoding="utf-8") as f:
        f.write(context)
        
    return {"context": context}

def retrieve_web_node(state: AgentState):
    print("--- SEARCHING THE WEB ---")
    context = web_search.invoke(state["query"])
    return {"context": context}

def generate_node(state: AgentState):
    print("--- GENERATING FINAL RESPONSE ---")
    query = state["query"]
    context = state.get("context", "")
    route = state.get("route", "CHAT")
    
    if route == "CHAT":
        prompt = f"""You are AxIn Help. The user said: "{query}". 
        Acknowledge them warmly and briefly, then ask how you can help them with the AxIn platform today.
        
        Answer:"""
        response = fast_generator_llm.invoke(prompt).content 
        
    elif route == "WEB":
        prompt = f"""You are AxIn Help. Answer the user's question using ONLY the provided live web context.
        
        Web Context: {context}
        Question: {query}
        Answer:"""
        response = fast_generator_llm.invoke(prompt).content
        
    else: # RAG Route
        # DYNAMIC PROMPT UPDATE: ChatGPT-style conversational verbosity and follow-ups
        prompt = f"""You are AxIn Help: an expert, friendly AI assistant for the AxIn platform. Answer the user's question clearly and professionally using ONLY the provided context. 
        
        CRITICAL INSTRUCTION: The context below is extracted from PDFs and may contain broken formatting. Read it carefully and extract the requested information.
        
        FORMATTING & TONE RULES:
        1. Start with a warm, explanatory paragraph setting the context for your answer.
        2. If the answer involves a process, instructions, multiple steps, or a list of items, you MUST break it down and format it using clean markdown bullet points or a numbered list.
        3. Follow up the bullet points with a brief concluding paragraph summarizing the value of this action or providing extra context.
        4. ALWAYS end your response with a helpful follow-up question related to their query to keep the conversation going (e.g., "Would you like me to explain how to export this data?", "Do you need help navigating to a different module?").
        5. If the context does not contain the answer, explicitly state: "I don't have enough information in the AxIn documentation to answer that. Kindly provide me with more information or ask another question."
        
        Context: 
        {context}
        
        Question: {query}
        Answer:"""
        response = smart_generator_llm.invoke(prompt).content
    
    return {"response": response}

# --- 4. BUILD THE GRAPH ---

def route_to_next(state: AgentState):
    return state["route"]

workflow = StateGraph(AgentState)
workflow.add_node("router", router_node)
workflow.add_node("retrieve_rag", retrieve_rag_node)
workflow.add_node("retrieve_web", retrieve_web_node)
workflow.add_node("generate", generate_node)

workflow.set_entry_point("router")
workflow.add_conditional_edges("router", route_to_next, {"RAG": "retrieve_rag", "WEB": "retrieve_web", "CHAT": "generate"})
workflow.add_edge("retrieve_rag", "generate")
workflow.add_edge("retrieve_web", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()