import os
import re
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Initialize OpenAI client with API key from environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load and chunk the text file
def load_and_chunk(filepath, chunk_size=800):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    paragraphs = text.split('\n\n')
    chunks = []
    metadata = []

    current_chapter = "Unknown Chapter"
    current_section = "Unknown Section"
    current_subsection = "Unknown Subsection"
    current_topic = "Unknown Topic"
    temp_para = ""

    for para in paragraphs:
        para = para.strip()

        if para.startswith('# '):
            current_chapter = para[2:].strip()
            continue
        if para.startswith('## '):
            current_section = para[3:].strip()
            continue
        if para.startswith('### '):
            current_subsection = para[4:].strip()
            continue
        if para.startswith('##### '):
            current_topic = para[6:].strip()
            continue
        if para.startswith('###### '):
            current_topic = para[7:].strip()
            continue

        if len(para) > 0:
            temp_para += para + " "
            if len(temp_para) >= chunk_size:
                chunks.append(temp_para.strip())
                metadata.append({
                    "chapter": current_chapter,
                    "section": current_section,
                    "subsection": current_subsection,
                    "topic": current_topic
                })
                temp_para = ""

    if temp_para:
        chunks.append(temp_para.strip())
        metadata.append({
            "chapter": current_chapter,
            "section": current_section,
            "subsection": current_subsection,
            "topic": current_topic
        })

    return chunks, metadata

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Normalize embeddings for cosine similarity
def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms

# Create or load FAISS index
def get_faiss_index(chunks, index_path="index_cosine.faiss", meta_path="chunks_cosine.pkl"):
    if os.path.exists(index_path) and os.path.exists(meta_path):
        print("Loading existing FAISS index and metadata...")
        index = faiss.read_index(index_path)
        with open(meta_path, 'rb') as f:
            stored_chunks = pickle.load(f)
        return stored_chunks, index

    print("Creating FAISS index from scratch...")
    embeddings = model.encode(chunks, convert_to_numpy=True)
    embeddings = normalize_embeddings(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    with open(meta_path, 'wb') as f:
        pickle.dump(chunks, f)
    return chunks, index

# Improved keyword extractor with synonyms
def extract_keywords(query):
    base_words = [word.lower() for word in re.findall(r'\w+', query) if len(word) > 3]
    
    synonyms = {
        "ukc": ["ukc", "under keel clearance", "clearance"],
        "depth": ["depth", "deep", "water depth"],
        "anchorage": ["anchorage", "anchor", "anchoring"],
        "watchkeeping": ["watchkeeping", "bridge watch", "watch"],
        "draft": ["draft", "draught"]
    }
    
    expanded = set()
    for word in base_words:
        expanded.add(word)
        for key, syns in synonyms.items():
            if word in syns:
                expanded.update(syns)

    return list(expanded)

# Soft keyword matching
def keyword_match(chunk, keywords):
    chunk_lower = chunk.lower()
    for keyword in keywords:
        if keyword in chunk_lower:
            return True
    return False

# Find relevant chunks with fallback
def find_relevant_chunks(query, chunks, index, top_k=8):
    query_embedding = model.encode([query], convert_to_numpy=True)
    query_embedding = normalize_embeddings(query_embedding)

    distances, indices = index.search(query_embedding, top_k)
    initial_results = [chunks[i] for i in indices[0]]

    important_words = extract_keywords(query)
    filtered_results = [chunk for chunk in initial_results if keyword_match(chunk, important_words)]

    if filtered_results:
        return filtered_results

    # Fallback: simple full-text contains
    brute_matches = [chunk for chunk in chunks if any(word in chunk.lower() for word in important_words)]
    if brute_matches:
        return brute_matches[:top_k]

    return initial_results  # final fallback

# Ask GPT using context
def ask_gpt(question, context):
    prompt = f"""
You are a helpful assistant. Based only on the information provided below, answer the user's question.

- Use your best judgement to infer an answer if relevant information is found.
- If the information is absolutely not present, then reply exactly: 'The answer is not available in the database.'

Context from database:
{context}

Question: {question}
Answer:
"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Initialize chatbot
def initialize_chatbot(file_path):
    print("Loading and chunking file...")
    chunks, metadata = load_and_chunk(file_path)
    print(f"Chunked into {len(chunks)} parts. Loading FAISS index...")
    chunks, index = get_faiss_index(chunks)
    print("Chatbot ready.")
    return chunks, metadata, index
