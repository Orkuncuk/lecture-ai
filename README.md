# 🎓 LectureAI: Smart Audio Transcription & Q&A Assistant

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://senin-app-linkin.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Groq API](https://img.shields.io/badge/API-Groq-orange.svg)](https://groq.com/)

LectureAI is an end-to-end web application designed to help students and professionals extract maximum value from audio and video lectures. It uses state-of-the-art AI models to transcribe audio, generate intelligent summaries, create structured PDF notes, and provide an interactive Q&A interface over the lecture material.

## ✨ Features

* **High-Accuracy Transcription:** Leverages `whisper-large-v3` (via Groq API) for fast and highly accurate multilingual transcription, specifically optimized for mixed-language (English/Turkish) academic content.
* **Smart Hallucination Filtering:** Includes custom algorithms to detect and drop Whisper hallucinations, loop repetitions, and language inconsistencies.
* **Instant PDF Generation:** Automatically compiles the transcript and an AI-generated executive summary into a neatly formatted, downloadable PDF note.
* **Interactive RAG Q&A:** Employs Retrieval-Augmented Generation (RAG) using `FAISS` and local embeddings to let users chat directly with their lecture notes and ask complex questions.

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **LLM & Audio API:** Groq (Whisper-large-v3, LLaMA-3/Mixtral for Q&A)
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **Embeddings:** `sentence-transformers` (paraphrase-multilingual-MiniLM-L12-v2)
* **Document Generation:** `fpdf2`

## 🚀 How It Works (Architecture)

1. **Upload & Transcribe:** The user uploads a media file (up to 25MB). The audio is processed through Groq's Whisper API.
2. **Quality Control Pipeline:** The raw transcript segments are evaluated. Low-confidence, nonsensical, or repetitive segments are automatically flagged or repaired.
3. **Vectorization:** The finalized transcript is chunked and embedded into a local FAISS index.
4. **Query & Retrieval:** When the user asks a question, the system retrieves the most relevant transcript chunks and feeds them to the LLM to generate a precise, context-aware answer.

## 💻 How to Run Locally (Step-by-Step Guide)

Anyone can clone and run this project on their local machine. Follow these instructions:

### Prerequisites
* Python 3.10 or higher installed on your system.
* A free Groq API Key. (You can get one quickly at [console.groq.com](https://console.groq.com/keys)).

### Installation Steps

**1. Clone the repository**
```bash
git clone [https://github.com/YOUR_USERNAME/lecture-ai.git](https://github.com/YOUR_USERNAME/lecture-ai.git)
cd lecture-ai
