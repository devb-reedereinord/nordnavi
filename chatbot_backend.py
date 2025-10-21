import os
import re
import uuid
import pickle
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# -------- Config --------
MODEL_NAME = os.getenv("NORDNAVI_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("NORDNAVI_EMBED_MODEL", "all-MiniLM-L6-v2")

# -------- Clients / Models --------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
embedder = SentenceTransformer(EMBED_MODEL)

# -------- Data structures --------
@dataclass
class Chunk:
    id: str
    text: str
    chapter: str
    section: str
    subsection: str
    topic: str

@dataclass
class KB:
    chunks: List[Chunk]
    faiss_index: Any
    embeddings: np.ndarray
    tfidf_vocab: Dict[str, int]
    tfidf_matrix: np.ndarray  # dense for simplicity
    id_to_idx: Dict[str, int]

# --------- Text utilities ---------
def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())

def _heading_level(line: str) -> int:
    m = re.match(r'^(#{1,6})\s+', line)
    return len(m.group(1)) if m else 0

# --------- Loader / chunker ---------
def load_and_chunk(filepath: str, chunk_size: int = 1400, overlap: int = 120) -> List[Chunk]:
    lines = Path(filepath).read_text(encoding="utf-8").splitlines()
    chapter = section = subsection = topic = "Unknown"
    buffer = []
    chunks = []

    def flush_buffer():
        nonlocal buffer, chunks, chapter, section, subsection, topic
        if not buffer:
            return
        text = _normalize(" ".join(buffer))
        # sliding windows inside the buffer for long sections
        start = 0
        while start < len(text):
            part = text[start:start+chunk_size]
            if not part:
                break
            chunks.append(Chunk(
                id=str(uuid.uuid4()),
                text=part,
                chapter=chapter, section=section, subsection=subsection, topic=topic
            ))
            if start + chunk_size >= len(text):
                break
            start = max(start + chunk_size - overlap, start + 1)
        buffer = []

    for raw in lines:
        lvl = _heading_level(raw)
        if lvl == 1:
            flush_buffer()
            chapter = raw.lstrip("# ").strip()
        elif lvl == 2:
            flush_buffer()
            section = raw.lstrip("# ").strip()
        elif lvl == 3:
            flush_buffer()
            subsection = raw.lstrip("# ").strip()
        elif lvl >= 4:
            flush_buffer()
            topic = raw.lstrip("# ").strip()
        else:
            if raw.strip() == "":
                buffer.append("\n")
            else:
                buffer.append(raw.strip() + " ")
    flush_buffer()
    return chunks

# --------- FAISS + TF-IDF ---------
def _normalize_embeddings(emb):
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    return emb / norms

def _build_faiss(embeddings: np.ndarray):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index

def _build_tfidf_matrix(chunks: List[Chunk]) -> Tuple[Dict[str,int], np.ndarray]:
    # simple TF-IDF (no external deps) — bag of words, lowercase, alnum tokens
    from collections import Counter
    docs_tokens = []
    vocab_set = set()
    for c in chunks:
        tokens = re.findall(r"[a-zA-Z0-9]{3,}", c.text.lower())
        docs_tokens.append(tokens)
        vocab_set.update(tokens)
    vocab = {t:i for i,t in enumerate(sorted(vocab_set))}
    N = len(chunks)
    V = len(vocab)
    df = np.zeros(V, dtype=np.float32)
    tf_counts = []
    for tokens in docs_tokens:
        cnt = Counter(tokens)
        tf_counts.append(cnt)
        for t in cnt:
            df[vocab[t]] += 1.0
    idf = np.log((N + 1) / (df + 1)) + 1  # smoothed
    # build dense TF-IDF (for small corpora this is fine)
    tfidf = np.zeros((N, V), dtype=np.float32)
    for i, cnt in enumerate(tf_counts):
        total = sum(cnt.values()) or 1
        for t, c in cnt.items():
            j = vocab[t]
            tfidf[i, j] = (c / total) * idf[j]
    # L2 normalize rows
    norms = np.linalg.norm(tfidf, axis=1, keepdims=True) + 1e-12
    tfidf = tfidf / norms
    return vocab, tfidf

# --------- Public API ---------
def initialize_chatbot(file_path: str) -> KB:
    chunks = load_and_chunk(file_path)
    texts = [c.text for c in chunks]
    emb = embedder.encode(texts, convert_to_numpy=True, batch_size=64, show_progress_bar=False)
    emb = _normalize_embeddings(emb)
    faiss_index = _build_faiss(emb)
    vocab, tfidf = _build_tfidf_matrix(chunks)
    id_to_idx = {c.id: i for i, c in enumerate(chunks)}
    return KB(chunks=chunks, faiss_index=faiss_index, embeddings=emb, tfidf_vocab=vocab, tfidf_matrix=tfidf, id_to_idx=id_to_idx)

def build_conversation_context(history_messages: List[Dict[str,str]]) -> str:
    out = []
    for m in history_messages:
        role = "User" if m["role"] == "user" else "Assistant"
        out.append(f"{role}: {m['content']}")
    return "\n".join(out)

def _keyword_boost(query: str) -> List[str]:
    base = [w.lower() for w in re.findall(r"[a-zA-Z0-9]{3,}", query)]
    synonyms = {
        "ukc": ["ukc", "under", "keel", "clearance"],
        "draft": ["draft", "draught"],
        "anchorage": ["anchorage", "anchor", "anchoring"],
        "pilotage": ["pilotage", "pilot", "harbor", "harbour"],
        "panama": ["panama", "canal", "neopanamax", "locks", "gatun"],
    }
    expanded = set(base)
    for w in base:
        for syns in synonyms.values():
            if w in syns:
                expanded.update(syns)
    return list(expanded)

def expand_scenario_query(query: str) -> str:
    """
    Expand situational queries with procedural and reasoning keywords
    so the retriever fetches action-based guidance, not just definitions.
    """
    keywords = [
        "action", "procedure", "response", "handling", "steps", "guideline",
        "responsibility", "what to do", "safety", "checklist", "risk",
        "prevention", "emergency", "instruction", "reporting"
    ]
    if any(word in query.lower() for word in ["if", "when", "during", "while", "happens", "situation", "scenario"]):
        query = query + " " + " ".join(keywords)
    return query

def retrieve_with_hybrid_search(kb: KB, query: str, conversation: str = "", top_k: int = 8) -> List[Dict[str, Any]]:
    # Compose query with recent conversation for better semantic match
    composed = (conversation + "\n" + query).strip() if conversation else query

    # Semantic
    q_emb = embedder.encode([composed], convert_to_numpy=True)
    q_emb = _normalize_embeddings(q_emb)
    D, I = kb.faiss_index.search(q_emb, top_k * 3)  # over-fetch
    sem_ids = [int(i) for i in I[0]]

    # Lexical TF-IDF
    tokens = re.findall(r"[a-zA-Z0-9]{3,}", composed.lower())
    vec = np.zeros(len(kb.tfidf_vocab), dtype=np.float32)
    for t in tokens:
        j = kb.tfidf_vocab.get(t)
        if j is not None:
            vec[j] += 1.0
    if np.linalg.norm(vec) > 1e-9:
        vec = vec / (np.linalg.norm(vec) + 1e-12)
    # cosine similarity with dense tfidf
    lex_scores = kb.tfidf_matrix @ vec  # shape (N,)

    # Combine (min-max scale both, then weighted)
    sem_scores = np.zeros(len(kb.chunks), dtype=np.float32)
    sem_scores[sem_ids] = D[0]
    # min-max
    def scale(x):
        mn, mx = float(x.min()), float(x.max())
        return (x - mn) / (mx - mn + 1e-9) if mx > mn else x*0
    s_sem = scale(sem_scores)
    s_lex = scale(lex_scores)
    w_sem, w_lex = 0.65, 0.35
    combo = w_sem * s_sem + w_lex * s_lex

    # Keyword filter (light)
    boosts = set(_keyword_boost(query))
    mask = np.zeros(len(kb.chunks), dtype=bool)
    for i, c in enumerate(kb.chunks):
        if any(b in c.text.lower() for b in boosts):
            mask[i] = True
    # rank
    idxs = np.argsort(-combo)
    ranked = [i for i in idxs if combo[i] > 0]
    if mask.any():
        ranked = [i for i in ranked if mask[i]] + [i for i in ranked if not mask[i]]
    top = ranked[:top_k]

    results = []
    for i in top:
        c = kb.chunks[i]
        results.append({
            "id": c.id,
            "chunk": c.text,
            "meta": {
                "chapter": c.chapter,
                "section": c.section,
                "subsection": c.subsection,
                "topic": c.topic
            }
        })
    return results

def _build_prompt(retrieved: List[Dict[str, Any]], query: str, strict: bool) -> str:
    ctx = "\n\n---\n\n".join([r["chunk"] for r in retrieved])
    mode = (
        "You are a company documentation assistant specializing in ship management and operations. "
        "Your role is to interpret user questions—including situational or 'what if' questions—"
        "based strictly on the company's documented procedures and best maritime practices."
    )
    policy = (
        "If the context does not directly describe the situation, reason through the most appropriate "
        "steps, responses, or preventive actions based on related context and logical inference. "
        "Always indicate when an inference or assumption is being made."
        if strict else
        "Prefer the context, but you may combine it with sound maritime reasoning. "
        "Do not contradict the context."
    )
    return f"""{mode}

{policy}

[Context begins]
{ctx}
[Context ends]

User question: {query}

Answer clearly, step-by-step if applicable, citing context where relevant.
"""

def ask_gpt_with_context(query: str, retrieved: List[Dict[str, Any]], strict: bool = True) -> str:
    prompt = _build_prompt(retrieved, query, strict)
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content

def get_source_metadata(kb: KB, chunk_id: str) -> Dict[str,str]:
    i = kb.id_to_idx.get(chunk_id, None)
    if i is None:
        return {"chapter":"Unknown","section":"Unknown","subsection":"Unknown","topic":"Unknown"}
    c = kb.chunks[i]
    return {"chapter": c.chapter, "section": c.section, "subsection": c.subsection, "topic": c.topic}



