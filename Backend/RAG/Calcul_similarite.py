import numpy as np
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from RAG.llama_index_rag import LlamaIndexRAGProcessor
import ast
import psycopg2
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def cosine_similarity(vec_a, vec_b):
    """Retourne la similarité cosinus entre deux vecteurs."""
    a = np.array(vec_a)
    b = np.array(vec_b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def hybrid_retrieval(query, top_k=5, alpha=0.5):
    """
    Combine BM25 (lexical) et similarité cosinus (sémantique) pour la recherche hybride.
    alpha = 0.5 -> équilibre entre les deux
    """
    # Connexion à la base
    conn = psycopg2.connect(
        host="localhost",
        port="5432",
        database="chatbotdb_2025",
        user="admin",
        password="docuBot111"
    )
    cur = conn.cursor()
    cur.execute("SELECT chunk_content, embedding FROM pdf_info;")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    chunks = [row[0] for row in rows]
    embeddings = [np.array(ast.literal_eval(row[1])) for row in rows]

    # --- Étape 1 : BM25 lexical ---
    tokenized_corpus = [word_tokenize(chunk.lower()) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = word_tokenize(query.lower())
    bm25_scores = np.array(bm25.get_scores(tokenized_query))

    # --- Étape 2 : Similarité cosinus ---
    Processor=LlamaIndexRAGProcessor()
    query_embedding = Processor.generate_embedding(query)
    cos_scores = np.array([cosine_similarity(query_embedding, emb) for emb in embeddings])

    # --- Étape 3 : Normalisation des scores ---
    def normalize(x):
        return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-9)

    bm25_norm = normalize(bm25_scores)
    cos_norm = normalize(cos_scores)

    # --- Étape 4 : Fusion pondérée ---
    hybrid_scores = alpha * cos_norm + (1 - alpha) * bm25_norm

    # --- Étape 5 : Top-k résultats ---
    top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
    top_chunks = [chunks[i] for i in top_indices]

    logger.info(f"✅ Hybrid retrieval a retourné {len(top_chunks)} passages.")
    return top_chunks
