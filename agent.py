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
    
    prompt = f"""You are a master routing AI. You must classify the user's query into exactly one of these three categories: RAG, WEB, or CHAT.
    
    Category Definitions:
    - RAG: The query asks about AxIn, ClickOPS, Reporting Hub, e-NOC, Site Watch, Site Specific View, or any internal platform documentation.
    - WEB: The query asks about live, real-time data, news, countries, current events, or weather.
    - CHAT: The query is a standard greeting (e.g., "Hi", "Hello"), casual conversation, or small talk.
    
    Examples:
    Query: "Hi" -> CHAT
    Query: "Hello there" -> CHAT
    Query: "What is AxIn?" -> RAG
    Query: "How does the Reporting Hub work?" -> RAG
    Query: "What does eNOC do?" -> RAG
    Query: "Who won the game last night?" -> WEB
    Query: "What is the temperature in Lahore?" -> WEB
    Query: "What is the current situation in Iran?" -> WEB
    
    User Query: "{query}"
    Classification (Respond with exactly one word):"""
    
    decision = router_llm.invoke(prompt).content.upper()
    
    if "RAG" in decision:
        route = "RAG"
    elif "WEB" in decision:
        route = "WEB"
    else:
        route = "CHAT"
        
    print(f"Decision: Route to {route}")
    return {"route": route}

def retrieve_rag_node(state: AgentState):
    print("--- RETRIEVING FROM LOCAL DATABASE ---")
    query = state["query"]
    query_lower = query.lower()
    
    docs = vector_store.similarity_search(query, k=8)
    
    modules = ["reporting hub", "clickops", "site watch", "e-noc", "site specific view"]
    target_module = next((m for m in modules if m in query_lower), None)
    
    if target_module:
        print(f"Boosting search for module: {target_module}")
        boosted_query = f"{target_module} {query} how to use navigate access steps login"
        priority_docs = vector_store.similarity_search(boosted_query, k=5)
        docs = priority_docs + docs 

    unique_contents = []
    seen = set()
    for d in docs:
        if d.page_content[:50] not in seen:
            unique_contents.append(d.page_content)
            seen.add(d.page_content[:50])

    context = "\n\n".join(unique_contents)
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
    
    # DYNAMIC MODEL SELECTION: Route determines which brain handles the generation
    if route == "CHAT":
        prompt = f"""You are AxIn Help: a helpful and professional AI assistant for the AxIn platform. 
        The user is engaging in casual conversation. Respond warmly and naturally. Do not mention that you lack documentation. Also, do not greet the user more than once if they have already greeted you. Just respond to the content of their message.
        
        User: {query}
        Answer:"""
        # Uses the 1B model
        response = fast_generator_llm.invoke(prompt).content 
        
    elif route == "WEB":
        prompt = f"""You are AxIn Help: a helpful and professional AI assistant for the AxIn platform. Answer the user's question using ONLY the provided live web context.
        
        Web Context: {context}
        
        Question: {query}
        Answer:"""
        # Uses the 1B model
        response = fast_generator_llm.invoke(prompt).content
        
    else: # RAG Route
        prompt = f"""You are AxIn Help: an expert AI assistant for the AxIn platform. Answer the user's question clearly and professionally using ONLY the provided context. 
        
        FORMATTING RULES:
        1. If the answer involves a process, instructions, multiple steps, or a list of items, you MUST break it down and format it using markdown bullet points or a numbered list.
        2. Never write long, comma-separated sentences for instructions. Convert them into clean, individual bullet points.
        3. If the context does not contain the answer, explicitly state: "I don't have enough information in the AxIn documentation to answer that. Kindly provide more information or ask a different question."
        
        Context: 
        {context}
        
        Question: {query}
        Answer:"""
        # Uses the 8B heavy model for accuracy
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