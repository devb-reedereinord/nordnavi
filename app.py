import json
import re
import uuid
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st

# ============== Optional: your backend imports ======================
# Expect these to be available in your project. If not, replace with
# your own implementations.
try:
    from chatbot_backend import (
        initialize_chatbot,
        build_conversation_context,
        retrieve_with_hybrid_search,
        ask_gpt_with_context,
        get_source_metadata,
    )
except Exception as e:
    st.warning(
        "Backend imports failed. Make sure 'chatbot_backend' with the required"
        " functions is available. Error: %s" % e
    )
    # Provide minimal fallbacks to keep the UI from crashing (optional)
    def initialize_chatbot():
        return {}

    def build_conversation_context(history_slice: List[Dict[str, str]]) -> str:
        return "\n".join([f"{m['role']}: {m['content']}" for m in history_slice])

    def retrieve_with_hybrid_search(kb, query: str, conversation: str, top_k: int = 8):
        return []

    def ask_gpt_with_context(query: str, conversation: str, retrieved: List[Dict[str, Any]]):
        return {"answer": "(No backend connected — this is a placeholder.)", "sources": []}

    def get_source_metadata(item: Dict[str, Any]):
        return {"title": "", "chapter": "", "section": "", "subsection": "", "topic": "", "path": ""}

# ======================== Config / Constants =========================
st.set_page_config(page_title="NordNavi Chat", page_icon="💬", layout="wide")

DATA_DIR = Path("data/chats")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ====================== User Identification ==========================

def get_user_id() -> str:
    """
    Returns a stable identifier for the current user.
    - If Streamlit Cloud auth is enabled, use email (if available via st.experimental_user).
    - Otherwise, ask the user once and keep it in session_state.
    """
    email = getattr(getattr(st, "experimental_user", None), "email", None)
    if email:
        st.session_state.user_id = email.lower().strip()
        return st.session_state.user_id

    if "user_id" not in st.session_state or not st.session_state.user_id:
        st.session_state.user_id = st.text_input(
            "Enter your email (used to keep your chats private):",
            placeholder="you@company.com",
            key="login_email",
        )
        if not st.session_state.user_id:
            st.stop()
        st.session_state.user_id = st.session_state.user_id.lower().strip()

    return st.session_state.user_id

# =================== Per-user storage helpers ========================

def _safe_filename(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)


def _user_chat_path(user_id: str) -> Path:
    return DATA_DIR / f"{_safe_filename(user_id)}.json"


def load_sessions(user_id: str) -> List[Dict[str, Any]]:
    p = _user_chat_path(user_id)
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []


def save_sessions(user_id: str, sessions: List[Dict[str, Any]]):
    p = _user_chat_path(user_id)
    with p.open("w", encoding="utf-8") as f:
        json.dump(sessions, f, ensure_ascii=False, indent=2)

# =================== Query Expansion (fix for NameError) =============

def expand_scenario_query(q: str) -> str:
    """
    Lightweight query expander for situational questions.
    Adds common IMS/SOP keywords and maps a few domain terms
    to synonyms/related terms to improve hybrid retrieval.
    """
    q_lower = q.lower()

    generic_boost = [
        "policy", "procedure", "procedures", "SOP", "checklist", "steps",
        "responsibilities", "reporting", "escalation", "mitigation",
        "risk assessment", "safety", "IMS", "compliance", "guidelines",
    ]

    expansions: List[str] = []
    def add(*words):
        expansions.extend(words)

    if any(w in q_lower for w in ["spill", "pollution", "sopep"]):
        add("oil spill", "pollution", "SOPEP", "response", "containment", "reporting")
    if any(w in q_lower for w in ["fire", "smoke", "hot work"]):
        add("fire", "emergency", "firefighting", "hot work", "permit to work", "muster", "drill")
    if any(w in q_lower for w in ["engine", "blackout", "breakdown"]):
        add("engine failure", "blackout", "propulsion", "ER procedures", "troubleshooting", "contingency")
    if any(w in q_lower for w in ["piracy", "security", "stowaway", "bmp"]):
        add("security", "ISPS", "SSP", "BMP5", "citadel", "report to authorities")
    if any(w in q_lower for w in ["collision", "allision", "grounding"]):
        add("navigational incident", "near miss", "reporting", "checklist", "master", "bridge team")
    if any(w in q_lower for w in ["cargo", "loading", "discharging", "isgott"]):
        add("ISGOTT", "cargo operations", "loading plan", "discharging plan", "ship/shore checklist")
    if any(w in q_lower for w in ["anchoring", "berthing", "unberthing", "mooring"]):
        add("mooring", "berthing", "unberthing", "anchoring", "risk assessment", "toolbox talk", "checklist")
    if any(w in q_lower for w in ["permit", "ptw", "work"]):
        add("permit to work", "PTW", "risk assessment", "JSA", "LOTO")

    if any(w in q_lower for w in ["if ", "when ", "during ", "while ", "happens", "in case"]):
        add("scenario", "what to do", "immediate actions", "contingency", "roles and responsibilities")

    expanded = " ".join([q] + generic_boost + list(dict.fromkeys(expansions)))
    return expanded

# ======================= App State Bootstrapping =====================

user_id = get_user_id()

if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = load_sessions(user_id)

if "current_chat" not in st.session_state:
    st.session_state.current_chat = 0 if st.session_state.chat_sessions else None

# Knowledge base / backend init
if "kb" not in st.session_state:
    st.session_state.kb = initialize_chatbot()

# ============================= UI ===================================

st.title("💬 NordNavi Chat")

# ---- Sidebar: Chats list & actions ----
with st.sidebar:
    st.subheader("Your Chats")

    # Create new chat
    if st.button("➕ New chat", use_container_width=True):
        new_chat = {
            "id": str(uuid.uuid4()),
            "title": "New Chat",
            "messages": [],  # list of {role: "user"|"assistant"|"system", content: str}
        }
        st.session_state.chat_sessions.append(new_chat)
        st.session_state.current_chat = len(st.session_state.chat_sessions) - 1
        save_sessions(user_id, st.session_state.chat_sessions)
        st.rerun()

    # List chats
    for idx, chat in enumerate(st.session_state.chat_sessions):
        is_current = (idx == st.session_state.current_chat)
        cols = st.columns([0.1, 0.75, 0.15])
        with cols[0]:
            if st.radio("", [""], index=0, key=f"sel_{chat['id']}", label_visibility="collapsed"):
                pass
        with cols[1]:
            if st.button(("🟢 " if is_current else "⚪ ") + chat.get("title", "Untitled"), key=f"btn_{chat['id']}"):
                st.session_state.current_chat = idx
                st.rerun()
        with cols[2]:
            if st.button("🗑️", key=f"del_{chat['id']}"):
                del st.session_state.chat_sessions[idx]
                if st.session_state.chat_sessions:
                    st.session_state.current_chat = min(idx, len(st.session_state.chat_sessions) - 1)
                else:
                    st.session_state.current_chat = None
                save_sessions(user_id, st.session_state.chat_sessions)
                st.rerun()

    # Rename current chat
    if st.session_state.current_chat is not None:
        curr = st.session_state.chat_sessions[st.session_state.current_chat]
        new_title = st.text_input("Rename chat", value=curr.get("title", "New Chat"))
        if new_title and new_title != curr.get("title"):
            curr["title"] = new_title
            save_sessions(user_id, st.session_state.chat_sessions)

# ---- Main: Conversation ----

if st.session_state.current_chat is None:
    st.info("Create a new chat to get started.")
    st.stop()

chat = st.session_state.chat_sessions[st.session_state.current_chat]
messages: List[Dict[str, str]] = chat.setdefault("messages", [])

# Render history
for m in messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"]) 

# Input
user_input = st.chat_input("Type your question…")

if user_input:
    # Append user message
    messages.append({"role": "user", "content": user_input})

    # Build short conversation context (last N turns)
    history_slice = messages[-8:]
    conversation_text = build_conversation_context(history_slice)

    # Expand query if it's a situational question
    expanded_query = expand_scenario_query(user_input)

    # Retrieve hybrid semantic + lexical context
    retrieved = retrieve_with_hybrid_search(
        st.session_state.kb,
        query=expanded_query,
        conversation=conversation_text,
        top_k=8,
    )

    # Ask LLM with retrieved context
    result = ask_gpt_with_context(
        query=user_input,
        conversation=conversation_text,
        retrieved=retrieved,
    )

    answer = result.get("answer", "(No answer)")
    sources = result.get("sources", [])

    # Append assistant message
    messages.append({"role": "assistant", "content": answer})

    # Persist
    save_sessions(user_id, st.session_state.chat_sessions)

    # Stream to UI
    with st.chat_message("assistant"):
        st.markdown(answer)
        # Optional: show sources as expandable list
        if sources:
            with st.expander("Sources"):
                for i, item in enumerate(sources, 1):
                    meta = get_source_metadata(item)
                    title = meta.get("title") or meta.get("path") or f"Source {i}"
                    chapter = meta.get("chapter", "")
                    section = meta.get("section", "")
                    subsection = meta.get("subsection", "")
                    topic = meta.get("topic", "")
                    st.markdown(
                        f"**{i}. {title}**\n\n"
                        f"- Chapter: {chapter}\n"
                        f"- Section: {section}\n"
                        f"- Subsection: {subsection}\n"
                        f"- Topic: {topic}"
                    )

    st.rerun()






