# AI Professional Medical Scribe

An end-to-end medical transcription, correction, and structured summarization pipeline designed for clinical use.

## 🚀 Features
- **High-Fidelity Transcription**: Powered by OpenAI Whisper v3 Large.
- **Medical Correction**: Dual-stage verification using BioBART and BioGPT to ensure clinical accuracy.
- **Structured SOAP Reports**: Generates granular clinical summaries including Subjective, Objective, Assessment, and Plan with CoT (Chain-of-Thought) reasoning.
- **Patient Simplification**: Automatically converts complex medical plans into simple 3-step patient instructions.
- **Knowledge Base**: Semantic history search using FAISS and clinical embeddings (Bio_ClinicalBERT).
- **Professional UI**: Modern Dark Mode Streamlit application with a clean, high-fidelity aesthetic.

## 🛠️ Tech Stack
- **Models**: Whisper v3, BioBART, BioGPT, SciSpacy, Medical T5.
- **Backend**: Python, PyTorch, Transformers, FAISS.
- **Frontend**: Streamlit (Dark Mode).

## 📂 Project Structure
- `app.py`: Streamlit User Interface.
- `run_pipeline.py`: Main CLI orchestration script.
- `medical_summarizer.py`: SOAP and Patient summary generation logic.
- `medical_correction.py`: Medical text verification pipeline.
- `ner_extraction.py`: Medical entity extraction using SciSpacy.
- `knowledge_base.py`: FAISS-based semantic search module.
- `train_whisper.py`: Audio preprocessing and transcription.

## 📖 Usage
1. **CLI**: Run `python run_pipeline.py` to process local audio.
2. **Web App**: Run `streamlit run app.py` for the interactive dashboard.

---
*Developed for Professional Clinical Use*
