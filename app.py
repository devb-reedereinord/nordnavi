import streamlit as st
from chatbot_backend import (
    initialize_chatbot,
    build_conversation_context,
    retrieve_with_hybrid_search,
    ask_gpt_with_context,
    get_source_metadata
)
import json
from pathlib import Path

st.set_page_config(page_title="📚 NordNavi+", page_icon="🤖", layout="wide")

# ---------- Persistent storage helpers ----------
DATA_DIR = Path(".nordnavi")
DATA_DIR.mkdir(exist_ok=True)
SESSIONS_PATH = DATA_DIR / "sessions.json"

def load_sessions():
    if SESSIONS_PATH.exists():
        try:
            return json.loads(SESSIONS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []

def save_sessions(chat_sessions):
    try:
        SESSIONS_PATH.write_text(json.dumps(chat_sessions, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        st.warning(f"Couldn't save chat history: {e}")

# ---------- Load knowledge base (cached) ----------
@st.cache_resource(show_spinner=False)
def load_kb():
    file_path = "databasebot.txt"
    return initialize_chatbot(file_path)

kb = load_kb()

# ---------- Session state ----------
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = load_sessions()

if "current_chat" not in st.session_state:
    st.session_state.current_chat = 0 if st.session_state.chat_sessions else None

# Create first chat on cold start
if st.session_state.current_chat is None:
    st.session_state.chat_sessions.append({"title": "New Chat", "messages": []})
    st.session_state.current_chat = 0

# ---------- Sidebar ----------
with st.sidebar:
    st.header("💬 Chats")
    cols = st.columns([1,1])
    with cols[0]:
        if st.button("➕ New Chat", use_container_width=True):
            st.session_state.chat_sessions.append({"title": "New Chat", "messages": []})
            st.session_state.current_chat = len(st.session_state.chat_sessions) - 1
            save_sessions(st.session_state.chat_sessions)
            st.experimental_rerun()
    with cols[1]:
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.chat_sessions = []
            st.session_state.current_chat = None
            try:
                SESSIONS_PATH.unlink(missing_ok=True)
            except Exception:
                pass
            st.rerun()

    # Chat list
    for idx, chat in enumerate(st.session_state.chat_sessions):
        is_active = (idx == st.session_state.current_chat)
        label = chat["title"] or "Untitled"
        if st.button(("▶ " if is_active else "• ") + label, key=f"chat_{idx}", use_container_width=True):
            st.session_state.current_chat = idx
            st.rerun()

    # Preferences
    st.markdown("---")
    st.caption("Preferences")
    st.session_state.setdefault("strict_mode", True)
    st.session_state.setdefault("max_history_turns", 12)
    st.session_state["strict_mode"] = st.toggle("Strictly answer from context", value=st.session_state["strict_mode"])
    st.session_state["max_history_turns"] = st.slider("History turns used for context", 4, 30, st.session_state["max_history_turns"])

st.title("📚 NordNavi+")
st.caption("Private RAG chatbot")

active = st.session_state.chat_sessions[st.session_state.current_chat]

# ---------- Replay messages ----------
for msg in active["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Sources for this answer"):
                for i, s in enumerate(msg["sources"], start=1):
                    meta = get_source_metadata(kb, s["id"])
                    st.markdown(f"**Source {i}** — {meta['chapter']} ▸ {meta['section']} ▸ {meta['subsection']} ▸ {meta['topic']}")
                    st.code(s["preview"])

# ---------- Chat input ----------
user_input = st.chat_input("Ask about your database...")

if user_input:
    # Append user message
    active["messages"].append({"role": "user", "content": user_input})
    # Title update
    if active["title"] == "New Chat":
        active["title"] = user_input[:40] + ("..." if len(user_input) > 40 else "")

    with st.spinner("🔎 Retrieving relevant context..."):
        # Build conversation-aware query
        history_slice = active["messages"][-(st.session_state["max_history_turns"]*2):]
        conversation_text = build_conversation_context(history_slice)
        # Hybrid retrieve
        retrieved = retrieve_with_hybrid_search(kb, query=user_input, conversation=conversation_text, top_k=8)

    with st.spinner("🤖 Thinking..."):
        # Ask model with safety rails
        answer = ask_gpt_with_context(
            query=user_input,
            retrieved=retrieved,
            strict=st.session_state["strict_mode"]
        )

    # Prepare pretty sources
    pretty_sources = [{
        "id": r["id"],
        "preview": r["chunk"][:800]
    } for r in retrieved]

    active["messages"].append({"role": "assistant", "content": answer, "sources": pretty_sources})

    # Persist
    save_sessions(st.session_state.chat_sessions)

    # Stream to UI
    with st.chat_message("assistant"):
        st.markdown(answer)
        if pretty_sources:
            with st.expander("📄 Sources for this answer"):
                for i, s in enumerate(pretty_sources, start=1):
                    meta = get_source_metadata(kb, s["id"])
                    st.markdown(f"**Source {i}** — {meta['chapter']} ▸ {meta['section']} ▸ {meta['subsection']} ▸ {meta['topic']}")
                    st.code(s["preview"])


