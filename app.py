import streamlit as st
import os
import json
import time
from datetime import datetime
import torch

# Import pipeline components
from train_whisper import check_device, preprocess_audio, transcribe_whisper, save_transcript
from medical_correction import MedicalCorrectionPipeline
from ner_extraction import MedicalNERPipeline
from medical_summarizer import MedicalSummarizer
from knowledge_base import MedicalKnowledgeBase, VECTOR_DB_PATH

# Page Configuration
st.set_page_config(
    page_title="AI Scribe - Professional Medical Pipeline",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Mode Styling
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    [data-testid="stSidebar"] {
        background-color: #1a1c24;
    }
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: #0e1117;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #0e1117;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #262730 !important;
    }
    /* Card-like containers for dark mode */
    div.stAlert {
        background-color: #1a1c24;
        border: 1px solid #30363d;
        color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize Session State
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = {
        "device": check_device(),
        "corrector": None,
        "ner": None,
        "summarizer": None,
        "kb": None
    }

def load_models():
    """Lazy loading of models to save memory in UI."""
    with st.spinner("📦 Initialization System Models (BioBART, BioGPT, MedGemma)..."):
        device = st.session_state.pipeline["device"]
        if not st.session_state.pipeline["corrector"]:
            st.session_state.pipeline["corrector"] = MedicalCorrectionPipeline(device)
        if not st.session_state.pipeline["ner"]:
            st.session_state.pipeline["ner"] = MedicalNERPipeline()
        if not st.session_state.pipeline["summarizer"]:
            st.session_state.pipeline["summarizer"] = MedicalSummarizer(device)
        if not st.session_state.pipeline["kb"]:
            st.session_state.pipeline["kb"] = MedicalKnowledgeBase(device)
    st.success("✅ All models loaded and ready on " + device.upper())

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3222/3222800.png", width=100)
    st.title("Medical Scribe")
    st.info(f"📡 Device: {st.session_state.pipeline['device'].upper()}")
    
    st.subheader("Control Panel")
    if st.button("🔄 Initialize/Reload Models", use_container_width=True):
        load_models()
        
    st.divider()
    st.caption("Developed for Professional Clinical Use")

# --- Main App ---
st.title("🏥 AI Professional Medical Scribe")
st.markdown("### End-to-End Transcription & Clinical Reasoning")

tabs = st.tabs(["🎙️ New Recording", "📄 Analysis Report", "🧠 Knowledge Base", "⚙️ Settings"])

# Tab 1: New Recording
with tabs[0]:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Upload Audio")
        audio_file = st.file_uploader("Choose a clinical recording (.mp4, .wav, .mp3, .m4a)", type=['mp4', 'wav', 'mp3', 'm4a'])
        
        if audio_file:
            st.audio(audio_file)
            if st.button("🚀 Start Professional Processing", type="primary"):
                if not st.session_state.pipeline["summarizer"]:
                    st.error("Please initialize models in the sidebar first!")
                else:
                    # Save uploader file temporarily
                    temp_path = os.path.join("/tmp", audio_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(audio_file.getbuffer())
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        # STAGE 1
                        status_text.text("🛠️ Preprocessing Audio...")
                        processed_path = preprocess_audio(temp_path)
                        progress_bar.progress(20)
                        
                        # STAGE 2
                        status_text.text("🎙️ Whisper v3 Transcription...")
                        raw_text = transcribe_whisper(processed_path, st.session_state.pipeline["device"])
                        progress_bar.progress(40)
                        
                        # STAGE 3
                        status_text.text("🧠 AI Medical Correction & Verification...")
                        final_text = st.session_state.pipeline["corrector"].process(raw_text)
                        progress_bar.progress(60)
                        
                        # STAGE 4
                        status_text.text("🔍 Extracting Medical Entities (SciSpacy)...")
                        entities = st.session_state.pipeline["ner"].extract_entities(final_text)
                        progress_bar.progress(75)
                        
                        # STAGE 5
                        status_text.text("📝 Generating Clinical SOAP Summary...")
                        soap_note = st.session_state.pipeline["summarizer"].generate_soap_note(final_text, entities)
                        
                        patient_summary = st.session_state.pipeline["summarizer"].generate_patient_summary(soap_note, entities)
                        progress_bar.progress(90)
                        
                        # STAGE 6
                        status_text.text("💾 Updating Knowledge Base...")
                        st.session_state.pipeline["kb"].add_consultation(final_text, entities, audio_file.name)
                        progress_bar.progress(100)
                        
                        # Store results for display
                        st.session_state.last_result = {
                            "transcript": final_text,
                            "soap": soap_note,
                            "patient": patient_summary,
                            "entities": entities
                        }
                        
                        st.success("✅ Medical processing complete!")
                        status_text.empty()
                        
                    except Exception as e:
                        st.error(f"💥 Pipeline Error: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

    with col2:
        st.subheader("Pipeline Info")
        st.write("- **ASR**: Whisper v3 Large")
        st.write("- **Reasoning**: MedGemma / BioGPT")
        st.write("- **NER**: SciSpacy (CDR)")
        st.write("- **KB**: FAISS Vector Index")

# Tab 2: Analysis Report
with tabs[1]:
    if "last_result" in st.session_state:
        res = st.session_state.last_result
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 👨‍⚕️ Clinical SOAP Note")
            st.info(res["soap"])
        with c2:
            st.markdown("#### 🤝 Patient-Friendly Summary")
            st.success(res["patient"])
            
        st.divider()
        st.markdown("#### 🔍 Extracted Medical Entities")
        e_col1, e_col2, e_col3 = st.columns(3)
        with e_col1:
            st.error(f"**Diseases**: {', '.join(res['entities'].get('DISEASES', []))}")
        with e_col2:
            meds = [f"{m['name']} ({m['dosage']})" for m in res['entities'].get('MEDICATIONS_WITH_DETAILS', [])]
            st.success(f"**Medications**: {', '.join(meds) if meds else 'None'}")
        with e_col3:
            st.warning(f"**Others**: {', '.join([str(x) for x in res['entities'].get('OTHER', [])])}")

        with st.expander("👁️ View Full Corrected Transcript"):
            st.write(res["transcript"])
    else:
        st.info("No active report. Process a recording in the 'New Recording' tab first.")

# Tab 3: Knowledge Base
with tabs[2]:
    st.subheader("Semantic Patient History Search")
    q = st.text_input("Enter clinical query (e.g. 'previous breast surgery', 'amoxicillin dosage')")
    
    if q and st.session_state.pipeline["kb"]:
        results = st.session_state.pipeline["kb"].search(q, top_k=3)
        if results:
            for r in results:
                with st.container():
                    st.markdown(f"**Match Clarity:** {r['similarity']:.3f}")
                    st.write(f"Source: {r['audio_source']} | Date: {r['timestamp'][:10]}")
                    st.markdown(f"> {r['text']}")
                    st.divider()
        else:
            st.warning("No matches found in the history.")
    elif not st.session_state.pipeline["kb"]:
        st.warning("Initialize models to search the Knowledge Base.")

# Tab 4: Settings
with tabs[3]:
    st.subheader("System Configuration")
    st.write(f"**Vector DB Path**: `{VECTOR_DB_PATH}`")
    st.write(f"**Models Cache**: `~/.cache/huggingface`")
    if st.button("🗑️ Clear Pipeline Cache"):
        st.cache_data.clear()
        st.success("Cache cleared.")
