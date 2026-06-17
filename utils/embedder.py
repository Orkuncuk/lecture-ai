import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Çok dilli model — TR/EN karışık metinler için ideal
_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_model = None  # Lazy load — ilk çağrıda yükle


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """
    Metni kelime bazında örtüşen chunk'lara böler.

    Args:
        text: Ham metin
        chunk_size: Her chunk'taki max kelime sayısı
        overlap: Ardışık chunk'lar arasındaki örtüşme (kelime)

    Returns:
        list[str]: Chunk listesi
    """
    # Fazla boşlukları temizle
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # overlap kadar geri git

    return [c for c in chunks if len(c.strip()) > 20]  # Çok kısa chunk'ları at


def build_index(chunks: list[str]) -> tuple[faiss.IndexFlatL2, np.ndarray]:
    """
    Chunk listesinden FAISS index oluşturur.

    Returns:
        (faiss_index, embeddings_array)
    """
    model = get_model()
    embeddings = model.encode(chunks, show_progress_bar=False, convert_to_numpy=True)
    embeddings = embeddings.astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index, embeddings


def search_index(
    query: str,
    index: faiss.IndexFlatL2,
    chunks: list[str],
    top_k: int = 4,
) -> list[str]:
    """
    Kullanıcı sorusuna en yakın chunk'ları getirir.

    Args:
        query: Kullanıcının sorusu
        index: FAISS index
        chunks: Orijinal metin parçaları
        top_k: Kaç chunk dönsün

    Returns:
        list[str]: En alakalı chunk'lar (skora göre sıralı)
    """
    model = get_model()
    q_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(q_emb, top_k)

    results = []
    for idx in indices[0]:
        if 0 <= idx < len(chunks):
            results.append(chunks[idx])

    return results


def embed_transcript(text: str) -> dict:
    """
    Tek fonksiyon ile chunk → embed → index pipeline'ı çalıştır.
    Streamlit session state'e kaydetmek için dict döner.

    Returns:
        dict: { "chunks": list, "index": faiss.Index, "embeddings": np.ndarray }
    """
    chunks = chunk_text(text)
    index, embeddings = build_index(chunks)
    return {
        "chunks": chunks,
        "index": index,
        "embeddings": embeddings,
    }