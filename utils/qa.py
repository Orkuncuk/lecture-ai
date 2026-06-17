import os
from groq import Groq
import faiss
from dotenv import load_dotenv

from utils.embedder import search_index

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

SYSTEM_PROMPT = """Sen bir üniversite ders asistanısın. 
Sana verilen ders notu bölümlerini kullanarak öğrencilerin sorularını yanıtlıyorsun.

Kurallar:
- Sadece verilen bağlamda olan bilgileri kullan
- Bağlamda yoksa "Bu bilgi ders notlarında geçmiyor" de
- Teknik terimleri hem Türkçe hem İngilizce açıkla (ders TR/EN karışık)
- Kısa, net ve öğrenciye yönelik yanıtlar ver
- Gerektiğinde madde madde listele"""


def answer_question(
    question: str,
    index: faiss.IndexFlatL2,
    chunks: list[str],
    chat_history: list[dict] = None,
    top_k: int = 4,
) -> str:
    """
    Kullanıcı sorusunu alır, ilgili chunk'ları bulur ve Groq LLaMA3 ile yanıtlar.

    Args:
        question: Öğrencinin sorusu
        index: FAISS vektör indeksi
        chunks: Ham metin parçaları
        chat_history: Önceki konuşmalar [{"role": ..., "content": ...}]
        top_k: Kaç chunk kullanılsın

    Returns:
        str: LLM'in yanıtı
    """
    # 1. En alakalı chunk'ları çek
    relevant_chunks = search_index(question, index, chunks, top_k=top_k)
    context = "\n\n---\n\n".join(relevant_chunks)

    # 2. Mesaj geçmişini hazırla
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if chat_history:
        # Son 6 mesajı al (context window taşmasın)
        messages.extend(chat_history[-6:])

    # 3. Bağlamı + soruyu ekle
    messages.append({
        "role": "user",
        "content": f"""Ders notu bölümleri:
{context}

Soru: {question}"""
    })

    # 4. Groq LLaMA3 ile yanıt al
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.3,      # Düşük temp → tutarlı, gerçekçi yanıtlar
        max_tokens=800,
    )

    return response.choices[0].message.content


def generate_summary(transcript_text: str) -> str:
    """
    Transkripsiyon metninden otomatik ders özeti üretir.
    PDF için kullanılır.
    """
    # Uzun metinleri kes (LLM context limiti)
    max_chars = 6000
    text_snippet = transcript_text[:max_chars]
    if len(transcript_text) > max_chars:
        text_snippet += "\n[metin kısaltıldı...]"

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "Sen bir üniversite ders asistanısın. "
                    "Verilen ders transkripsiyonundan kısa ve öz bir özet çıkar. "
                    "Anahtar kavramları, önemli terimleri ve ana başlıkları vurgula. "
                    "Türkçe yaz, teknik terimler İngilizce kalabilir. "
                    "Maksimum 200 kelime."
                ),
            },
            {
                "role": "user",
                "content": f"Şu ders transkriptini özetle:\n\n{text_snippet}",
            },
        ],
        temperature=0.4,
        max_tokens=400,
    )

    return response.choices[0].message.content