import streamlit as st
from chatbot_backend import initialize_chatbot, find_relevant_chunks, ask_gpt

# Load the chatbot system on first run
@st.cache_resource
def load_chatbot():
    file_path = "databasebot.txt"
    chunks, metadata, index = initialize_chatbot(file_path)
    return chunks, metadata, index

st.set_page_config(page_title="📚 Private Chatbot", page_icon="🤖")
st.title("📚 NordNavi")

# Load resources
with st.spinner("Loading database and building chatbot..."):
    chunks, metadata, index = load_chatbot()

# Session state initialization
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = []  # List of chats
if "current_chat" not in st.session_state:
    st.session_state.current_chat = None  # Active chat index

# Sidebar for chat navigation and history controls
with st.sidebar:
    st.header("💬 Chat History")
    
    # New Chat button - starts a new conversation
    if st.button("➕ New Chat"):
        st.session_state.chat_sessions.append({
            "title": "New Chat",
            "messages": []
        })
        st.session_state.current_chat = len(st.session_state.chat_sessions) - 1
        st.experimental_rerun()
    
    # Clear Chat History button - wipes all stored conversations
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_sessions = []
        st.session_state.current_chat = None
        st.experimental_rerun()
    
    # List previous chats for selection
    for idx, chat in enumerate(st.session_state.chat_sessions):
        button_label = chat["title"] if chat["title"] != "" else "Untitled Chat"
        if st.button(button_label, key=f"chat_{idx}"):
            st.session_state.current_chat = idx
            st.experimental_rerun()

# If no chat exists yet, create one
if st.session_state.current_chat is None and not st.session_state.chat_sessions:
    st.session_state.chat_sessions.append({
        "title": "New Chat",
        "messages": []
    })
    st.session_state.current_chat = 0

# Get active chat
active_chat = st.session_state.chat_sessions[st.session_state.current_chat]

# Replay previous messages with per-message source display
for message in active_chat["messages"]:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        else:
            st.markdown(f"**Bot:** {message['content']}")
            if "sources" in message and message["sources"]:
                # Show primary source fully
                best_chunk = message["sources"][0]
                try:
                    meta_idx = chunks.index(best_chunk)
                    meta = metadata[meta_idx]
                    with st.expander("📄 Primary Source for this answer"):
                        st.markdown(f"""
📚 **Chapter:** {meta['chapter']}  
📑 **Section:** {meta['section']}  
🧩 **Subsection:** {meta['subsection']}  
🧷 **Topic:** {meta['topic']}  

> {best_chunk}
                        """)
                except ValueError:
                    st.markdown("> Primary source metadata not found.")
                # List other sources in brief if available
                if len(message["sources"]) > 1:
                    with st.expander("📄 Other Related Sources (Brief)"):
                        for idx2, chunk in enumerate(message["sources"][1:], start=2):
                            try:
                                meta_idx = chunks.index(chunk)
                                meta = metadata[meta_idx]
                                st.markdown(f"- **Source {idx2}:** {meta['chapter']} ➔ {meta['section']} ➔ {meta['subsection']} ➔ {meta['topic']}")
                            except ValueError:
                                st.markdown(f"- Source {idx2}: Metadata not found.")

# Chat input area
user_input = st.chat_input("Ask a question based on the database:")

if user_input:
    # Save user message in current chat session
    active_chat["messages"].append({"role": "user", "content": user_input})
    
    # Update chat title based on the first message if it is still "New Chat"
    if active_chat["title"] == "New Chat":
        active_chat["title"] = user_input[:40] + "..." if len(user_input) > 40 else user_input
    # Optionally, you can generate a dynamic title using GPT once a few messages are exchanged.
    elif len(active_chat["messages"]) == 3:
        try:
            auto_title = ask_gpt("Summarize this conversation into a short title:", user_input)
            active_chat["title"] = auto_title.strip()
        except Exception:
            pass

    with st.chat_message("user"):
        st.markdown(f"**You:** {user_input}")

    with st.spinner("Thinking..."):
        # Build full conversation context for better responses (last 6 exchanges)
        max_turns = 6  # 6 exchanges (user + assistant pairs)
        relevant_messages = active_chat["messages"][-max_turns*2:]
        conversation_context = ""
        for msg in relevant_messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            conversation_context += f"{role}: {msg['content']}\n"

        # Retrieve relevant chunks based on full context and latest user input
        relevant_chunks = find_relevant_chunks(conversation_context + user_input, chunks, index)
        context = "\n---\n".join(relevant_chunks)

        # Generate GPT answer using the context
        response = ask_gpt(user_input, context)

    # Save assistant response along with associated sources (only if answer is found)
    active_chat["messages"].append({
        "role": "assistant",
        "content": response,
        "sources": relevant_chunks if "not available" not in response.lower() else []
    })

    # Display assistant response and the corresponding sources
    with st.chat_message("assistant"):
        st.markdown(f"**Bot:** {response}")
        if "not available" not in response.lower() and relevant_chunks:
            best_chunk = relevant_chunks[0]
            try:
                meta_idx = chunks.index(best_chunk)
                meta = metadata[meta_idx]
                with st.expander("📄 Primary Source for this answer"):
                    st.markdown(f"""
📚 **Chapter:** {meta['chapter']}  
📑 **Section:** {meta['section']}  
🧩 **Subsection:** {meta['subsection']}  
🧷 **Topic:** {meta['topic']}  

> {best_chunk}
                    """)
            except ValueError:
                st.markdown("> Primary source metadata not found.")
            if len(relevant_chunks) > 1:
                with st.expander("📄 Other Related Sources (Brief)"):
                    for idx2, chunk in enumerate(relevant_chunks[1:], start=2):
                        try:
                            meta_idx = chunks.index(chunk)
                            meta = metadata[meta_idx]
                            st.markdown(f"- **Source {idx2}:** {meta['chapter']} ➔ {meta['section']} ➔ {meta['subsection']} ➔ {meta['topic']}")
                        except ValueError:
                            st.markdown(f"- Source {idx2}: Metadata not found.")
        else:
            st.info("No relevant sources found in the database.")