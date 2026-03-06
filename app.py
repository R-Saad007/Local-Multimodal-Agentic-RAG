import streamlit as st
from agent import app  
import time
import html

# --- 1. UI CONFIGURATION & LOGO INJECTION ---
st.set_page_config(page_title="AxIn Actionable Intelligence", page_icon="axin_logo.png", layout="wide")

st.markdown("""
    <style>
    /* Kill the massive default padding at the top of Streamlit's main container */
    .block-container {
        padding-top: 1rem !important; 
    }

    /* Dynamic Pill Styling - Hugging the top left corner */
    .dynamic-status-pill {
        display: inline-flex;
        background-color: #1E1E1E;
        color: #FFFFFF;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 13px;
        font-family: 'Segoe UI', sans-serif;
        font-weight: 500;
        align-items: center;
        gap: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-top: 30px; /* Pulls it right to the top edge */
        margin-bottom: 15px;
    }
   
    .status-dot {
        height: 10px;
        width: 10px;
        background-color: #00E676;
        border-radius: 50%;
        display: inline-block;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 230, 118, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(0, 230, 118, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 230, 118, 0); }
    }

    .user-msg-container {
        display: flex;
        flex-direction: row-reverse;
        align-items: flex-start;
        gap: 12px;
        margin-bottom: 24px;
    }
   
    .user-avatar {
        background-color: #2e2e2e;
        border-radius: 50%;
        width: 35px;
        height: 35px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
   
    .user-msg {
        background-color: #2B2D31;
        color: white;
        padding: 14px 20px;
        border-radius: 20px 4px 20px 20px;
        max-width: 75%;
        font-family: 'Segoe UI', sans-serif;
        font-size: 15px;
        line-height: 1.5;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
   
    .stChatMessage { border-radius: 15px; margin-bottom: 15px; }
    .stStatusWidget { border: none !important; box-shadow: none !important; }
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Render the dynamic pill absolutely first so it sits at the peak of the layout
st.markdown("""
    <div class="dynamic-status-pill">
        <span class="status-dot"></span>
        Ollama Connected | Llama 3.1 (8B)
    </div>
""", unsafe_allow_html=True)

st.title("AxIn Help")
st.caption("Proprietary Actionable Intelligence Assistant")

BOT_AVATAR = "axin_logo.png"

if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {"Starter": []}
    st.session_state.current_session = "Starter"

with st.sidebar:
    if st.button("➕ New Chat", use_container_width=True):
        new_session_name = f"New Chat {len(st.session_state.chat_sessions) + 1}"
        st.session_state.chat_sessions[new_session_name] = []
        st.session_state.current_session = new_session_name
        st.rerun()

    st.write("### Chats")
   
    for session_name in reversed(list(st.session_state.chat_sessions.keys())):
        btn_type = "primary" if session_name == st.session_state.current_session else "secondary"
        if st.button(session_name, use_container_width=True, type=btn_type):
            st.session_state.current_session = session_name
            st.rerun()

active_messages = st.session_state.chat_sessions[st.session_state.current_session]

for message in active_messages:
    if message["role"] == "user":
        safe_text = html.escape(message["content"])
        user_html = f"""
        <div class="user-msg-container">
            <div class="user-avatar">👤</div>
            <div class="user-msg">{safe_text}</div>
        </div>
        """
        st.markdown(user_html, unsafe_allow_html=True)
    else:
        try:
            with st.chat_message("assistant", avatar=BOT_AVATAR):
                st.markdown(message["content"])
        except Exception:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(message["content"])

if prompt := st.chat_input("How may I help you today?"):
   
    current_session = st.session_state.current_session
    if len(st.session_state.chat_sessions[current_session]) == 0:
        new_title = prompt[:25] + ("..." if len(prompt) > 25 else "")
        base_title = new_title
        counter = 1
        while new_title in st.session_state.chat_sessions:
            new_title = f"{base_title} ({counter})"
            counter += 1
           
        st.session_state.chat_sessions[new_title] = st.session_state.chat_sessions.pop(current_session)
        st.session_state.current_session = new_title

    st.session_state.chat_sessions[st.session_state.current_session].append({"role": "user", "content": prompt})
   
    safe_prompt = html.escape(prompt)
    st.markdown(f"""
        <div class="user-msg-container">
            <div class="user-avatar">👤</div>
            <div class="user-msg">{safe_prompt}</div>
        </div>
    """, unsafe_allow_html=True)

    try:
        bot_container = st.chat_message("assistant", avatar=BOT_AVATAR)
    except Exception:
        bot_container = st.chat_message("assistant", avatar="🤖")

    with bot_container:
        try:
            # We start the status box with the initial label
            with st.status("🔍 Analyzing user intent...", expanded=True) as status:
                result_state = None
                
                for output in app.stream({"query": prompt}):
                    for node_name, node_state in output.items():
                        result_state = node_state 
                        
                        # Dynamically change the title of the status box based on the route
                        if node_name == "router":
                            route = node_state.get("route", "CHAT")
                            if route == "RAG":
                                status.update(label="📚 Searching AxIn documentation...")
                            elif route == "WEB":
                                status.update(label="🌐 Searching the live web...")
                            elif route == "CHAT":
                                status.update(label="💬 Engaging in conversation...")
                                
                        # Change the title again when it moves to the generation phase
                        elif node_name in ["retrieve_rag", "retrieve_web"]:
                            status.update(label="✍️ Generating response...")

                # Final update when the graph finishes
                status.update(label="✅ Response Ready", state="complete", expanded=False)

            full_response = result_state["response"]
           
            placeholder = st.empty()
            typed_text = ""
            for char in full_response:
                typed_text += char
                placeholder.markdown(typed_text + "▌")
                time.sleep(0.005)
            placeholder.markdown(full_response)
           
            st.session_state.chat_sessions[st.session_state.current_session].append({"role": "assistant", "content": full_response})

            st.rerun()

        except Exception as e:
            st.error(f"System Error: {str(e)}")