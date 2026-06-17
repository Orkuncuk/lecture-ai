import os
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ─────────────────────────────────────────────────────────────────────────────
# Dile göre Whisper prompt'u
# ─────────────────────────────────────────────────────────────────────────────
_PROMPTS = {
    "tr": (
        "Bu bir üniversite dersidir. Hoca Türkçe konuşuyor, "
        "ama teknik terimler İngilizce olabilir. "
        "Noktalama işaretlerini doğru koy."
    ),
    "en": (
        "This is a university lecture in English. "
        "It may contain technical and academic terminology. "
        "Use correct punctuation."
    ),
    "mixed": (
        "Bu bir üniversite dersidir. "
        "Hoca Türkçe ve İngilizce karışık konuşabilir. "
        "Teknik terimler, formüller ve akademik kavramlar içerebilir. "
        "Noktalama işaretlerini doğru koy."
    ),
}

# Streamlit'teki seçim -> Whisper'a verilecek dil kodu (None = otomatik algıla)
_LANG_CODES = {"tr": "tr", "en": "en", "mixed": None}

# ─────────────────────────────────────────────────────────────────────────────
# Eşik değerler
# ─────────────────────────────────────────────────────────────────────────────
_ALLOWED_CHARS = re.compile(
    r"[a-zA-ZçğıöşüÇĞİÖŞÜ0-9\s\.,!?;:'\"\-\(\)\[\]/%&@#\*\+=<>_°²³€$₺]"
)

_GARBAGE_RATIO_THRESHOLD = 0.15      # karakter setine bakarak "anlamsız metin" eşiği
_NO_SPEECH_THRESHOLD = 0.6           # Whisper'ın "burada konuşma yok" güveni
_COMPRESSION_RATIO_THRESHOLD = 2.4   # tekrar/loop halüsinasyonu sinyali (Whisper'ın kendi eşiği)
_SHORT_WORD_RATIO_THRESHOLD = 0.40   # "ünlüsüz iskelet" tipi halüsinasyon
_MIN_WORDS_FOR_SHORT_CHECK = 4
_LOW_CONFIDENCE_THRESHOLD = -0.8     # avg_logprob bundan düşükse "belirsiz, kontrol et"
_OVERLAP_THRESHOLD = 0.3             # onarımda zaman örtüşme oranı

# [YENİ] no_speech tek başına yetmez; aynı anda düşük confidence de istiyoruz
_NO_SPEECH_LOGPROB_GATE = -0.5     # avg_logprob bundan kötü VE no_speech yüksekse drop

# [YENİ] Repair çıktısı için ayrı, daha sıkı eşik
_REPAIR_SHORT_WORD_THRESHOLD = 0.22
# Tek harf + rakam karışık dizi: "g H 4 edisi", "y 5 X bil" tarzı
_REPAIR_GARBAGE_PATTERN = re.compile(r"\b[a-zA-ZçğıöşüÇĞİÖŞÜ]\s+[a-zA-ZçğıöşüÇĞİÖŞÜ0-9]\b")

# [YENİ] Loop hallucination eşiği
_LOOP_SIM_THRESHOLD = 0.85    # ardışık iki segment bu kadar benzese loop say


# ─────────────────────────────────────────────────────────────────────────────
# Dil tutarlılığı için Türkçe'ye özgü karakter seti
# ─────────────────────────────────────────────────────────────────────────────
_TURKISH_SPECIFIC = frozenset("çğışöüÇĞİŞÖÜ")
_TURKISH_CHAR_RATIO_THRESHOLD = 0.015   # Harflerin %1.5'i Türkçe özel → Türkçe metin say
_DOMINANT_LANG_RATIO = 1.5              # Bir dil diğerinden 1.5x fazlaysa "baskın"
_MIN_SEGS_FOR_DOMINANT = 3             # Baskın dil tespiti için min. referans segment sayısı


def _seg_get(seg, key, default=0.0):
    """Segment hem dict hem obje olarak gelebilir, ikisini de destekle."""
    return seg.get(key, default) if isinstance(seg, dict) else getattr(seg, key, default)


def _garbage_ratio(text: str) -> float:
    stripped = text.strip()
    if not stripped:
        return 1.0
    allowed = len(_ALLOWED_CHARS.findall(stripped))
    return 1 - (allowed / len(stripped))


def _short_word_ratio(text: str) -> float:
    words = text.split()
    if len(words) < _MIN_WORDS_FOR_SHORT_CHECK:
        return 0.0
    short = sum(1 for w in words if len(re.sub(r"[^\wçğıöşüÇĞİÖŞÜ]", "", w)) <= 2)
    return short / len(words)


def _is_low_quality(text: str) -> bool:
    return (
        _garbage_ratio(text) > _GARBAGE_RATIO_THRESHOLD
        or _short_word_ratio(text) > _SHORT_WORD_RATIO_THRESHOLD
    )


# ─────────────────────────────────────────────────────────────────────────────
# Loop (Tekrar) Halüsinasyon Dedektörü
# ─────────────────────────────────────────────────────────────────────────────
def _normalize_for_repeat(text: str) -> str:
    """İki segmenti karşılaştırırken büyük/küçük, noktalama, boşluk normalize."""
    return re.sub(r"[^\wçğıöşüÇĞİÖŞÜ]+", " ", text.lower()).strip()


def _similarity(a: str, b: str) -> float:
    """Karakter düzeyinde basit benzerlik. difflib istemiyoruz, hızlı olsun."""
    if not a or not b:
        return 0.0
    a, b = _normalize_for_repeat(a), _normalize_for_repeat(b)
    if a == b:
        return 1.0
    # Jaccard-ish over word sets (hızlı ve "aynı içerik tekrarı" için yeterli)
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _flag_loops(metrics: list[dict]) -> None:
    """
    İki ardışık segment çok benziyorsa ikincisini (ve sonraki tekrarları)
    loop halüsinasyonu olarak işaretle. metrics in-place değişir.
    """
    for i in range(1, len(metrics)):
        if metrics[i]["is_bad"] or not metrics[i]["text"]:
            continue
        if _similarity(metrics[i - 1]["text"], metrics[i]["text"]) >= _LOOP_SIM_THRESHOLD:
            metrics[i]["is_bad"] = True
            metrics[i]["reason"] = "loop"


# ─────────────────────────────────────────────────────────────────────────────
# Onarım (Repair) Katı Filtresi
# ─────────────────────────────────────────────────────────────────────────────
def _is_repair_garbage(text: str) -> bool:
    """
    Repair sonucu için _is_low_quality'den daha sıkı kontrol. Repair'in
    döndürdüğü 'parayda g H 4 edisi' tarzı halüsinasyonları yakalar.
    """
    if not text or len(text.split()) < _MIN_WORDS_FOR_SHORT_CHECK:
        return True
    if _garbage_ratio(text) > _GARBAGE_RATIO_THRESHOLD:
        return True
    if _short_word_ratio(text) > _REPAIR_SHORT_WORD_THRESHOLD:
        return True
    # "g H 4 edisi" gibi tek harf+rakam karışımı kalıpları
    if len(_REPAIR_GARBAGE_PATTERN.findall(text)) >= 2:
        return True
    # Kelime ortasında BÜYÜK harf (Whisper'ın hallüsine ettiği "tokenı çözememe" izi)
    mid_caps = sum(1 for w in text.split() if re.search(r"[a-zçğıöşü][A-ZÇĞİÖŞÜ]", w))
    if mid_caps >= 2:
        return True
    return False


def _segment_metrics(seg) -> dict:
    """
    Bir segmentin tüm kalite sinyallerini tek yerde toplar.
    """
    text = _seg_get(seg, "text", "").strip()
    no_speech = _seg_get(seg, "no_speech_prob", 0.0) or 0.0
    compression = _seg_get(seg, "compression_ratio", 0.0) or 0.0
    avg_logprob = _seg_get(seg, "avg_logprob", 0.0) or 0.0
    garbage = _garbage_ratio(text)
    short_word = _short_word_ratio(text)

    if not text:
        reason = "empty"
    # [DEĞİŞTİ]: no_speech tek başına kill değil. Sadece confidence de düşükse drop.
    elif no_speech >= _NO_SPEECH_THRESHOLD and avg_logprob < _NO_SPEECH_LOGPROB_GATE:
        reason = "no_speech"
    elif compression >= _COMPRESSION_RATIO_THRESHOLD:
        reason = "compression"
    elif garbage > _GARBAGE_RATIO_THRESHOLD:
        reason = "garbage_chars"
    elif short_word > _SHORT_WORD_RATIO_THRESHOLD:
        reason = "short_words"
    else:
        reason = None

    return {
        "text": text,
        "no_speech_prob": round(no_speech, 3),
        "compression_ratio": round(compression, 3),
        "avg_logprob": round(avg_logprob, 3),
        "garbage_ratio": round(garbage, 3),
        "short_word_ratio": round(short_word, 3),
        "is_bad": reason is not None,
        "reason": reason,
    }


def _resolve(language: str):
    language = language if language in _PROMPTS else "mixed"
    return _LANG_CODES[language], _PROMPTS[language]


def _call_whisper(file_tuple, language_code, prompt, temperature=0.0):
    return client.audio.transcriptions.create(
        file=file_tuple,
        model="whisper-large-v3",
        response_format="verbose_json",
        language=language_code,
        prompt=prompt,
        temperature=temperature,
    )


def _candidate_from_segments(primary_seg, repair_segments) -> str | None:
    p_start = _seg_get(primary_seg, "start", 0.0)
    p_end = _seg_get(primary_seg, "end", p_start)
    p_dur = max(p_end - p_start, 0.01)

    overlapping = []
    for rseg in repair_segments:
        r_start = _seg_get(rseg, "start", 0.0)
        r_end = _seg_get(rseg, "end", r_start)
        overlap = min(p_end, r_end) - max(p_start, r_start)
        if overlap / p_dur >= _OVERLAP_THRESHOLD:
            overlapping.append(rseg)

    if not overlapping:
        return None

    overlapping.sort(key=lambda s: _seg_get(s, "start", 0.0))
    candidate_text = " ".join(_seg_get(s, "text", "").strip() for s in overlapping).strip()
    return candidate_text or None


def format_timestamp(seconds: float) -> str:
    seconds = int(seconds or 0)
    return f"{seconds // 60}:{seconds % 60:02d}"


def _turkish_char_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c in _TURKISH_SPECIFIC) / len(letters)


def _is_turkish_text(text: str) -> bool:
    return _turkish_char_ratio(text) > _TURKISH_CHAR_RATIO_THRESHOLD


def _dominant_language(segments: list[dict]) -> str | None:
    ref_segs = [
        s for s in segments
        if s["status"] in ("ok", "uncertain") and s.get("text", "").strip()
    ]
    if len(ref_segs) < _MIN_SEGS_FOR_DOMINANT:
        return None

    turkish_n = sum(1 for s in ref_segs if _is_turkish_text(s["text"]))
    english_n = len(ref_segs) - turkish_n

    if turkish_n >= english_n * _DOMINANT_LANG_RATIO:
        return "tr"
    if english_n >= turkish_n * _DOMINANT_LANG_RATIO:
        return "en"
    return None


def _language_consistency_filter(all_segments: list[dict]) -> list[dict]:
    dominant = _dominant_language(all_segments)
    if dominant is None:
        return all_segments

    for seg in all_segments:
        if seg["status"] != "repaired":
            continue
        seg_turkish = _is_turkish_text(seg["text"])
        mismatch = (dominant == "en" and seg_turkish) or \
                   (dominant == "tr" and not seg_turkish)
        if mismatch:
            seg["status"] = "dropped"
            seg["included"] = False
            seg["debug"] = {
                **seg["debug"],
                "reason": (seg["debug"].get("reason") or "") + "+lang_mismatch",
            }

    return all_segments


def _repair_language(primary_lang_choice: str, detected_lang: str | None = None) -> str:
    if detected_lang in ("tr", "en"):
        return detected_lang
    if primary_lang_choice == "tr":
        return "en"
    return "tr"


def rebuild_transcript(segments: list[dict]) -> str:
    parts = [s["text"] for s in segments if s.get("included") and s.get("text", "").strip()]
    return re.sub(r"\s{2,}", " ", " ".join(parts).strip())


def _build_transcript(file_tuple, primary, primary_lang_choice) -> dict:
    segments = getattr(primary, "segments", None)

    if not segments:
        return {
            "text": primary.text,
            "language": getattr(primary, "language", "unknown"),
            "duration": getattr(primary, "duration", 0.0),
            "segments": [],
        }

    metrics = [_segment_metrics(seg) for seg in segments]
    
    # Ardışık loop'ları burada bayrakla işaretliyoruz
    _flag_loops(metrics)                           
    
    bad_indices = {i for i, m in enumerate(metrics) if m["is_bad"]}

    # Sadece "mixed" modda repair dene. tr/en'de kullanıcının seçimine güven.
    repair_segments = []
    if bad_indices and primary_lang_choice == "mixed":
        detected_lang = getattr(primary, "language", None)
        repair_lang = _repair_language(primary_lang_choice, detected_lang)
        _, prompt = _resolve(repair_lang)
        try:
            repair = _call_whisper(file_tuple, _LANG_CODES[repair_lang], prompt, temperature=0.2)
            repair_segments = getattr(repair, "segments", []) or []
        except Exception:
            repair_segments = []

    all_segments = []
    for i, seg in enumerate(segments):
        start = _seg_get(seg, "start", 0.0)
        end = _seg_get(seg, "end", start)
        m = metrics[i]
        text = m["text"]

        if i not in bad_indices:
            status = "uncertain" if (m["avg_logprob"] < _LOW_CONFIDENCE_THRESHOLD and text) else "ok"
            all_segments.append({
                "index": i, "start": start, "end": end,
                "text": text, "included": True, "status": status, "debug": m,
            })
            continue

        # [DEĞİŞTİ]: Loop ya da tr/en modda olduğumuz için repair yoksa direkt dropped
        if m["reason"] == "loop" or not repair_segments:
            all_segments.append({
                "index": i, "start": start, "end": end,
                "text": text, "included": False, "status": "dropped", "debug": m,
            })
            continue

        candidate = _candidate_from_segments(seg, repair_segments)
        
        if candidate and not _is_repair_garbage(candidate):     
            all_segments.append({
                "index": i, "start": start, "end": end,
                "text": candidate, "included": True, "status": "repaired", "debug": m,
            })
        else:
            all_segments.append({
                "index": i, "start": start, "end": end,
                "text": text, "included": False, "status": "dropped", "debug": m,
            })

    if primary_lang_choice == "mixed":
        all_segments = _language_consistency_filter(all_segments)

    return {
        "text": rebuild_transcript(all_segments) or primary.text,
        "language": getattr(primary, "language", "unknown"),
        "duration": getattr(primary, "duration", 0.0),
        "segments": all_segments,
    }


def transcribe_audio(file_path: str, language: str = "mixed") -> dict:
    lang_code, prompt = _resolve(language)

    with open(file_path, "rb") as audio_file:
        file_bytes = audio_file.read()

    file_tuple = (os.path.basename(file_path), file_bytes)
    primary = _call_whisper(file_tuple, lang_code, prompt)
    return _build_transcript(file_tuple, primary, language)


def transcribe_bytes(file_bytes: bytes, filename: str, language: str = "mixed") -> dict:
    lang_code, prompt = _resolve(language)

    file_tuple = (filename, file_bytes)
    primary = _call_whisper(file_tuple, lang_code, prompt)
    return _build_transcript(file_tuple, primary, language)
