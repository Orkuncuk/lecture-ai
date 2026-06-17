import os
import streamlit as st

from utils.transcribe import transcribe_bytes, rebuild_transcript, format_timestamp
from utils.embedder import embed_transcript
from utils.pdf_gen import get_pdf_bytes
from utils.qa import answer_question, generate_summary

# ── Sayfa Ayarları ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LectureAI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title { font-size: 2rem; font-weight: 700; margin-bottom: 0; }
    .sub-title  { color: #666; font-size: 1rem; margin-bottom: 1.5rem; }
    .status-box {
        background: #f0f4ff; border-left: 4px solid #4060A0;
        padding: 0.8rem 1rem; border-radius: 4px; margin: 1rem 0;
        color: #1a1a2e;
    }
    .status-box b { color: #1a1a2e; }
    .chat-user, .chat-bot {
        border-radius: 12px; padding: 0.6rem 1rem; margin: 0.3rem 0; color: #1a1a2e;
    }
    .chat-user { background: #e8f0fe; text-align: right; }
    .chat-bot  { background: #f8f9fa; }
    .seg-meta  { color: #888; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)

# ── Başlık ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🎓 LectureAI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Ders sesini yükle → PDF al → Sorularını sor</p>', unsafe_allow_html=True)

# ── API Key Kontrolü ──────────────────────────────────────────────────────────
groq_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
if not groq_key:
    st.error("⚠️  GROQ_API_KEY bulunamadı. `.env` dosyana veya Streamlit secrets'a ekle.")
    st.code('export GROQ_API_KEY="gsk_..."', language="bash")
    st.stop()
os.environ["GROQ_API_KEY"] = groq_key

# ── Session State Başlat ──────────────────────────────────────────────────────
_DEFAULTS = {
    "transcript": None,       # Ham metin (segments'ten türetilir)
    "segments": [],            # Her segment: index, start, end, text, included, status, debug
    "meta": {},                 # Dil, süre, dosya adı
    "index_data": None,         # FAISS index + chunks
    "chat_history": [],         # Sohbet geçmişi
    "summary": None,            # LLM özeti
}
for key, default in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = (default.copy() if isinstance(default, (dict, list)) else default)


# ── Yardımcı Fonksiyonlar ──────────────────────────────────────────────────────
def _reembed_after_edit() -> None:
    """Segmentler değiştiğinde transkripti yeniden oluşturur ve indeksi günceller."""
    st.session_state.transcript = rebuild_transcript(st.session_state.segments)
    with st.spinner("🔍 Vektör indeksi güncelleniyor..."):
        st.session_state.index_data = embed_transcript(st.session_state.transcript)
    st.session_state.chat_history = []
    st.session_state.summary = None


def _apply_segment_text(seg_index: int, new_text: str, new_status: str | None = None) -> None:
    """Bir segmentin metnini günceller, transkripte dahil eder ve yeniden indeksler."""
    new_text = new_text.strip()
    if not new_text:
        return
    for seg in st.session_state.segments:
        if seg["index"] == seg_index:
            seg["text"] = new_text
            seg["included"] = True
            if new_status:
                seg["status"] = new_status
            break
    _reembed_after_edit()
    st.rerun()


def render_status_box(meta: dict, transcript: str) -> None:
    dur = meta.get("duration", 0)
    dur_str = f"{int(dur // 60)}dk {int(dur % 60)}sn" if dur else "—"
    st.markdown(f"""
    <div class="status-box">
        📌 <b>{meta.get('title', '')}</b><br>
        🌐 Dil: {meta.get('language', '').upper()} &nbsp;|&nbsp; ⏱️ Süre: {dur_str} &nbsp;|&nbsp;
        📝 {len(transcript.split())} kelime
    </div>
    """, unsafe_allow_html=True)


def render_transcript_quality_notes(segments: list) -> None:
    """Düzeltilen / silinen / belirsiz bölümleri gösterir; sil/şüpheli için düzenle+ekle imkanı sunar."""
    repaired = [s for s in segments if s["status"] == "repaired"]
    dropped = [s for s in segments if s["status"] == "dropped"]
    uncertain = [s for s in segments if s["status"] == "uncertain"]

    if repaired:
        st.success(f"🔧 {len(repaired)} bölüm, farklı dil denenerek otomatik olarak düzeltildi.")

    if dropped:
        st.info(
            f"ℹ️ {len(dropped)} bölüm hiçbir dil denemesinde anlaşılır çıkmadığı için temizlendi. "
            "Aşağıdan ne söylendiğini biliyorsan düzenleyip transkripte ekleyebilirsin."
        )
        with st.expander(f"🗑️ Silinen {len(dropped)} bölümü gör / ekle"):
            for seg in dropped:
                st.markdown(
                    f'<span class="seg-meta">⏱️ {format_timestamp(seg["start"])} - '
                    f'{format_timestamp(seg["end"])} '
                    f'(Whisper\'ın anlamadığı ham çıktı — referans olarak gösteriliyor)</span>',
                    unsafe_allow_html=True,
                )
                edited = st.text_area(
                    "Düzenle ve ekle",
                    value=seg["text"],
                    key=f"dropped_edit_{seg['index']}",
                    label_visibility="collapsed",
                )
                if st.button("➕ Transkripte Ekle", key=f"dropped_add_{seg['index']}"):
                    _apply_segment_text(seg["index"], edited, new_status="manual")
                st.divider()

    if uncertain:
        with st.expander(f"⚠️ {len(uncertain)} bölüm düşük güvenli — kontrol et / düzelt"):
            st.caption(
                "Bu bölümler halüsinasyon değil, ama Whisper kendinden emin değil. "
                "Belirtilen zaman aralığına gidip dinleyerek metni düzeltip güncelleyebilirsin."
            )
            for seg in uncertain:
                st.markdown(
                    f'<span class="seg-meta">⏱️ {format_timestamp(seg["start"])} - '
                    f'{format_timestamp(seg["end"])}</span>',
                    unsafe_allow_html=True,
                )
                edited = st.text_area(
                    "Düzenle ve güncelle",
                    value=seg["text"],
                    key=f"uncertain_edit_{seg['index']}",
                    label_visibility="collapsed",
                )
                if st.button("✏️ Güncelle", key=f"uncertain_update_{seg['index']}"):
                    _apply_segment_text(seg["index"], edited)
                st.divider()


def render_debug_panel(segments: list) -> None:
    """Eşik ayarlarını tune etmek için her segmentin ham kalite sinyallerini gösterir."""
    if not segments:
        return
    with st.expander("🔬 Debug: segment sinyalleri"):
        st.caption(
            "compression_ratio yüksekse (≥2.4) tekrar/loop halüsinasyonu, "
            "no_speech_prob yüksekse (≥0.6) konuşma yok, avg_logprob çok düşükse "
            "(<-0.8) Whisper kendinden emin değil demektir."
        )
        rows = []
        for seg in segments:
            d = seg["debug"]
            rows.append({
                "Zaman": f"{format_timestamp(seg['start'])}-{format_timestamp(seg['end'])}",
                "Durum": seg["status"],
                "Sebep": d["reason"] or "-",
                "no_speech": d["no_speech_prob"],
                "compression": d["compression_ratio"],
                "avg_logprob": d["avg_logprob"],
                "garbage": d["garbage_ratio"],
                "short_word": d["short_word_ratio"],
                "Metin": (seg["text"][:60] + "…") if len(seg["text"]) > 60 else seg["text"],
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)


# ── 3 Sekme ───────────────────────────────────────────────────────────────────
tab_upload, tab_pdf, tab_qa = st.tabs(["📤 Yükle & Transkribe", "📄 PDF İndir", "💬 Soru Sor"])


# ════════════════════════════════════════════════════════════════════════════════
# SEKME 1 — YÜKLE
# ════════════════════════════════════════════════════════════════════════════════
with tab_upload:
    st.subheader("Ders Kaydını Yükle")
    st.caption("Desteklenen formatlar: mp3, mp4, wav, m4a, webm, ogg (max 25 MB)")

    uploaded_file = st.file_uploader(
        "Ses veya video dosyası seç",
        type=["mp3", "mp4", "wav", "m4a", "webm", "ogg"],
        label_visibility="collapsed",
    )

    lecture_title = st.text_input(
        "Ders başlığı (opsiyonel)",
        placeholder="örn. Veri Yapıları — Hafta 3: Binary Search Tree",
    )

    lecture_lang = st.selectbox(
        "Ders dili",
        options=["mixed", "tr", "en"],
        format_func=lambda x: {
            "mixed": "🌐 Türkçe + İngilizce karışık (otomatik algıla)",
            "tr": "🇹🇷 Türkçe",
            "en": "🇬🇧 İngilizce",
        }[x],
        help=(
            "Dersin ağırlıklı dilini seçmek transkripsiyon kalitesini artırır. "
            "**Ders büyük oranda İngilizce ise 'mixed' yerine doğrudan "
            "'İngilizce' seçmen daha doğru sonuç verir** — Whisper, 'mixed' "
            "modda bazı bölümleri yanlışlıkla Türkçe sanıp o bölüm için "
            "anlamsız Türkçe metin üretebiliyor (halüsinasyon). Dersin "
            "ağırlıklı Türkçe olduğu durumlarda da aynı mantıkla 'Türkçe' "
            "seçmek daha iyi sonuç verir; 'mixed' sadece gerçekten dengeli "
            "karışık derslerde tercih edilmeli."
        ),
    )

    if uploaded_file:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.info(f"📁 **{uploaded_file.name}** — {file_size_mb:.1f} MB")

        start_btn = st.button("🚀 Transkribe Et", type="primary")

        if start_btn:
            with st.spinner("🎙️ Ses tanıma yapılıyor (Whisper large-v3)..."):
                try:
                    result = transcribe_bytes(uploaded_file.read(), uploaded_file.name, language=lecture_lang)
                except Exception as e:
                    st.error(f"Transkripsiyon hatası: {e}")
                    st.stop()

            st.session_state.segments = result["segments"]
            st.session_state.transcript = result["text"]
            st.session_state.meta = {
                "title": lecture_title or uploaded_file.name,
                "language": result["language"],
                "duration": result["duration"],
                "filename": uploaded_file.name,
            }
            st.session_state.chat_history = []
            st.session_state.summary = None

            with st.spinner("🔍 Vektör indeksi oluşturuluyor..."):
                try:
                    st.session_state.index_data = embed_transcript(st.session_state.transcript)
                except Exception as e:
                    st.error(f"İndeksleme hatası: {e}")
                    st.stop()

            st.success("✅ Transkripsiyon tamamlandı!")

    # Sonuç varsa göster
    if st.session_state.transcript:
        render_status_box(st.session_state.meta, st.session_state.transcript)
        render_transcript_quality_notes(st.session_state.segments)
        render_debug_panel(st.session_state.segments)

        with st.expander("📝 Transkripsiyon Önizlemesi"):
            preview = st.session_state.transcript[:2000]
            suffix = "..." if len(st.session_state.transcript) > 2000 else ""
            st.write(preview + suffix)
    else:
        st.info("👆 Bir dosya yükle ve 'Transkribe Et' butonuna bas.")


# ════════════════════════════════════════════════════════════════════════════════
# SEKME 2 — PDF
# ════════════════════════════════════════════════════════════════════════════════
with tab_pdf:
    st.subheader("PDF Oluştur ve İndir")

    if not st.session_state.transcript:
        st.warning("⚠️ Önce **Yükle** sekmesinden bir ses dosyası transkribe et.")
    else:
        st.caption("PDF; transkripsiyon + yapay zeka özeti içerir.")

        col1, col2 = st.columns([1, 1])
        with col1:
            include_summary = st.checkbox("🤖 Otomatik özet ekle (LLM)", value=True)
        with col2:
            custom_title = st.text_input(
                "PDF başlığı",
                value=st.session_state.meta.get("title", "Ders Notu"),
            )

        if st.button("📄 PDF Oluştur", type="primary"):
            if include_summary:
                with st.spinner("✍️ Özet üretiliyor..."):
                    try:
                        st.session_state.summary = generate_summary(st.session_state.transcript)
                    except Exception as e:
                        st.warning(f"Özet üretilemedi: {e}")

            with st.spinner("📄 PDF oluşturuluyor..."):
                try:
                    pdf_bytes = get_pdf_bytes(
                        transcript_text=st.session_state.transcript,
                        title=custom_title,
                        language=st.session_state.meta.get("language", "tr"),
                        duration=st.session_state.meta.get("duration", 0),
                        summary=st.session_state.summary,
                    )
                except Exception as e:
                    st.error(f"PDF hatası: {e}")
                    st.stop()

            safe_name = custom_title.replace(" ", "_").replace("/", "-")
            st.download_button(
                label="⬇️ PDF'i İndir",
                data=pdf_bytes,
                file_name=f"{safe_name}.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True,
            )

        if st.session_state.summary:
            with st.expander("📋 Özet Önizlemesi"):
                st.write(st.session_state.summary)


# ════════════════════════════════════════════════════════════════════════════════
# SEKME 3 — SORU SOR
# ════════════════════════════════════════════════════════════════════════════════
with tab_qa:
    st.subheader("Ders Notlarına Soru Sor")

    if not st.session_state.index_data:
        st.warning("⚠️ Önce **Yükle** sekmesinden bir ses dosyası transkribe et.")
    else:
        st.caption("Sorularını Türkçe veya İngilizce sorabilirsin.")

        for msg in st.session_state.chat_history:
            css_class = "chat-user" if msg["role"] == "user" else "chat-bot"
            icon = "🧑" if msg["role"] == "user" else "🤖"
            st.markdown(f'<div class="{css_class}">{icon} {msg["content"]}</div>', unsafe_allow_html=True)

        with st.form("qa_form", clear_on_submit=True):
            question = st.text_input(
                "Sorunuz",
                placeholder="örn. Binary Search Tree'nin zaman karmaşıklığı nedir?",
                label_visibility="collapsed",
            )
            col1, col2 = st.columns([3, 1])
            with col1:
                submitted = st.form_submit_button("Sor →", type="primary", use_container_width=True)
            with col2:
                clear_btn = st.form_submit_button("🗑️ Temizle", use_container_width=True)

        if clear_btn:
            st.session_state.chat_history = []
            st.rerun()

        if submitted and question.strip():
            with st.spinner("🤔 Yanıt üretiliyor..."):
                try:
                    answer = answer_question(
                        question=question,
                        index=st.session_state.index_data["index"],
                        chunks=st.session_state.index_data["chunks"],
                        chat_history=st.session_state.chat_history,
                    )
                except Exception as e:
                    st.error(f"Q&A hatası: {e}")
                    st.stop()

            st.session_state.chat_history.append({"role": "user", "content": question})
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.rerun()