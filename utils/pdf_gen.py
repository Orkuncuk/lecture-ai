import os
import re
from fpdf import FPDF
from datetime import datetime

# Proje köküne göre fonts/ klasörü
FONT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fonts")
FONT_REGULAR = os.path.join(FONT_DIR, "DejaVuSans.ttf")
FONT_BOLD = os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf")


class LecturePDF(FPDF):
    """Ders notları için modern ve estetik tasarımlı PDF sınıfı (Türkçe karakter destekli)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Türkçe karakter desteği için DejaVu Unicode fontları yüklenir
        self.add_font("DejaVu", "", FONT_REGULAR)
        self.add_font("DejaVu", "B", FONT_BOLD)
        self.add_font("DejaVu", "I", FONT_REGULAR)

        # Renk Paleti Tanımları (Modern Slate & Royal Blue Arayüzü)
        self.color_primary = (30, 41, 59)      # Slate-800 (Ana metinler/Başlıklar)
        self.color_accent = (37, 99, 235)      # Blue-600 (Vurgu rengi)
        self.color_text = (71, 85, 105)        # Slate-600 (Gövde metni)
        self.color_light_bg = (248, 250, 252)  # Slate-50 (Kart arka planları)
        self.color_border = (226, 232, 240)    # Slate-200 (Bölücüler ve sınırlar)

    def header(self):
        # Üst bilgi alanını sadece ilk sayfa hariç diğer sayfalarda göster
        if self.page_no() > 1:
            self.set_font("DejaVu", "", 8)
            self.set_text_color(148, 163, 184) # Slate-400
            
            # Sol kısım: Sabit etiket | Sağ kısım: Platform adı
            self.cell(90, 8, "Ders Notu Özeti & Transkripsiyon", align="L")
            self.cell(90, 8, "LectureAI", align="R")
            self.ln(8)
            
            # Çok ince ve estetik bir üst bölücü çizgi
            self.set_draw_color(*self.color_border)
            self.set_line_width(0.2)
            self.line(15, self.get_y(), 195, self.get_y())
            self.ln(6)

    def footer(self):
        # Sayfa altı sınır çizgisi
        self.set_y(-18)
        self.set_draw_color(*self.color_border)
        self.set_line_width(0.2)
        self.line(15, self.get_y(), 195, self.get_y())
        self.ln(3)

        # Sayfa numarası ve marka
        self.set_font("DejaVu", "I", 8)
        self.set_text_color(148, 163, 184)
        self.cell(90, 10, "LectureAI — Yapay Zeka Ders Asistanı", align="L")
        self.cell(90, 10, f"Sayfa {self.page_no()}", align="R")


def clean_text(text: str) -> str:
    """PDF'e yazılmadan önce metindeki gereksiz boşlukları temizler."""
    if not text:
        return ""
    text = re.sub(r" {2,}", " ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


def split_into_paragraphs(text: str, sentences_per_para: int = 5) -> list[str]:
    """Uzun transkripsiyonları okunabilir ideal paragraflara böler."""
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-ZÇĞİÖŞÜa-z])', text)
    paragraphs = []
    for i in range(0, len(sentences), sentences_per_para):
        chunk = " ".join(sentences[i:i + sentences_per_para])
        if chunk.strip():
            paragraphs.append(chunk.strip())
    return paragraphs


def draw_section_header(pdf: LecturePDF, title: str):
    """Bölüm başlıklarının yanına modern dikey vurgu çizgisi çizer."""
    pdf.ln(4)
    current_y = pdf.get_y()
    
    # Sayfa sonu kontrolü
    if current_y > 250:
        pdf.add_page()
        current_y = pdf.get_y()

    # Vurgu Çubuğu (Dikey mavi belirteç)
    pdf.set_fill_color(*pdf.color_accent)
    pdf.rect(15, current_y + 1, 3, 6, "F")
    
    # Başlık Metni
    pdf.set_font("DejaVu", "B", 13)
    pdf.set_text_color(*pdf.color_primary)
    pdf.set_x(21)
    pdf.cell(0, 8, title, ln=1)
    
    # Başlık altındaki zarif yatay çizgi
    pdf.set_draw_color(*pdf.color_border)
    pdf.set_line_width(0.4)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(5)


def generate_pdf(
    transcript_text: str,
    output_path: str,
    title: str = "Ders Notu",
    language: str = "tr",
    duration: float = 0.0,
    summary: str = None,
) -> str:
    """
    Transkripsiyon metninden modern, kurumsal kalitede bir PDF raporu üretir.
    """
    pdf = LecturePDF()
    pdf.set_margins(15, 15, 15)  # İdeal sayfa kenar boşlukları (Sol, Üst, Sağ)
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # ── BAŞLIK ALANI ──────────────────────────────────────────────
    pdf.set_font("DejaVu", "B", 20)
    pdf.set_text_color(*pdf.color_primary)
    # Büyük ve okunaklı ders başlığı
    pdf.multi_cell(0, 9, clean_text(title))
    pdf.ln(4)

    # ── METADATA BİLGİ KARTLARI (BADGES) ──────────────────────────
    # Kartların arka planı ve çerçeve renkleri ayarlanır
    pdf.set_fill_color(*pdf.color_light_bg)
    pdf.set_draw_color(*pdf.color_border)
    pdf.set_line_width(0.2)
    pdf.set_text_color(*pdf.color_text)
    pdf.set_font("DejaVu", "", 8.5)

    # 1. Kart: Tarih
    date_str = f"  Tarih: {datetime.now().strftime('%d.%m.%Y')}"
    pdf.cell(56, 8, date_str, border=1, fill=True, align="L")
    pdf.cell(4, 8, "", ln=0) # Ara boşluk
    
    # 2. Kart: Dil
    lang_label = "Türkçe/İngilizce" if language in ("tr", "en") else language.upper()
    lang_str = f"  Dil: {lang_label}"
    pdf.cell(56, 8, lang_str, border=1, fill=True, align="L")
    pdf.cell(4, 8, "", ln=0) # Ara boşluk
    
    # 3. Kart: Süre
    dur_min = f"  Süre: {int(duration // 60)}dk {int(duration % 60)}sn" if duration else "  Süre: Bilinmiyor"
    pdf.cell(60, 8, dur_min, border=1, fill=True, align="L")
    pdf.ln(14)

    # ── ÖZET ALANI (Alıntı/Callout Kutusu Tasarımı) ────────────────
    if summary:
        draw_section_header(pdf, "Yapay Zeka Özeti")
        
        # Özet paragraflarını temizle ve böl
        summary_paragraphs = clean_text(summary).split("\n")
        pdf.set_font("DejaVu", "I", 10)
        pdf.set_text_color(*pdf.color_primary)
        
        for para in summary_paragraphs:
            if not para.strip():
                continue
            
            # Dinamik yan çizgi çizimi için başlangıç koordinatı
            y_start = pdf.get_y()
            
            # İçerik yazımı (Sol taraftan 6mm içe kaydırılarak alıntı hissi verilir)
            pdf.set_x(21)
            pdf.multi_cell(174, 6.5, para)
            
            # Yazım sonrası koordinat
            y_end = pdf.get_y()
            
            # Alıntı Bloğu Sol Vurgu Çizgisi (Her paragrafın soluna dikey çizgi)
            pdf.set_fill_color(*pdf.color_accent)
            pdf.rect(15, y_start, 1.5, y_end - y_start - 1.5, "F")
            pdf.ln(3)
        pdf.ln(4)

    # ── TAM TRANSKRİPSİYON ALANI ──────────────────────────────────
    draw_section_header(pdf, "Tam Transkripsiyon")
    
    paragraphs = split_into_paragraphs(transcript_text)
    pdf.set_font("DejaVu", "", 10)
    pdf.set_text_color(*pdf.color_text)

    for para in paragraphs:
        # Metinlerin hizalaması iki yana yaslı (Justified) yapılarak kitap şıklığı elde edilir
        pdf.multi_cell(0, 6.5, clean_text(para), align="J")
        # Paragraflar arası ideal boşluk
        pdf.ln(4)

    # Dosyayı kaydet ve yolu döndür
    pdf.output(output_path)
    return output_path


def get_pdf_bytes(
    transcript_text: str,
    title: str = "Ders Notu",
    language: str = "tr",
    duration: float = 0.0,
    summary: str = None,
) -> bytes:
    """
    PDF belgesini diske kalıcı olarak yazmadan bellek (bytes) biçiminde döner.
    Streamlit download_button entegrasyonu için optimize edilmiştir.
    """
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        generate_pdf(transcript_text, tmp_path, title, language, duration, summary)
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)