import os
import torch
from train_whisper import find_audio_file, preprocess_audio, transcribe_whisper, check_device, save_transcript
from medical_correction import MedicalCorrectionPipeline
from ner_extraction import MedicalNERPipeline
from knowledge_base import MedicalKnowledgeBase, VECTOR_DB_PATH
from medical_summarizer import MedicalSummarizer

# --- Configuration ---
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    print("🚀 --- Integrated Medical Transcription, Correction & Summarization Pipeline --- 🚀")
    device = check_device()
    
    # Search for audio in the current folder
    audio_file = find_audio_file(OUTPUT_DIR)
    
    if not audio_file:
        print(f"❌ No audio files found in {OUTPUT_DIR}")
        return

    print(f"📂 Found Audio: {audio_file}")
    
    try:
        # Step 1: Preprocess Audio
        processed_audio = preprocess_audio(audio_file)
        
        # Step 2: Transcribe using Whisper v3
        print("\n--- STAGE 1: WHISPER TRANSCRIPTION ---")
        raw_transcript = transcribe_whisper(processed_audio, device)
        print("✅ Raw Transcription Complete.")
        
        # Step 3: Medical Correction System (The Two-Stage System)
        print("\n--- STAGE 2: MEDICAL CORRECTION PIPELINE ---")
        correction_system = MedicalCorrectionPipeline(device)
        final_transcript = correction_system.process(raw_transcript)
        
        # Step 4: Medical NER (SciSpacy)
        print("\n--- STAGE 3: MEDICAL ENTITY EXTRACTION ---")
        ner_system = MedicalNERPipeline()
        entities = ner_system.extract_entities(final_transcript)
        
        # New: Step 5: Final Medical Summarization (Professional + Patient)
        print("\n--- STAGE 4: MEDICAL SUMMARIZATION (SOAP + PATIENT) ---")
        summarizer = MedicalSummarizer(device)
        soap_note = summarizer.generate_soap_note(final_transcript, entities)
        patient_summary = summarizer.generate_patient_summary(soap_note, entities)
        
        # Step 6: Update Knowledge Base
        print("\n--- STAGE 5: UPDATING KNOWLEDGE BASE ---")
        kb = MedicalKnowledgeBase(device)
        kb.add_consultation(final_transcript, entities, os.path.basename(audio_file))
        
        # Display Final Results
        print("\n" + "="*50)
        print("📋 FINAL CLINICAL REPORT PREVIEW")
        print("="*50)
        print(f"\n👨‍⚕️ [SOAP NOTE]\n{soap_note[:1000]}...")
        print(f"\n🤝 [PATIENT SUMMARY]\n{patient_summary[:500]}...")
        
        print("\n--- MEDICAL ENTITIES ---")
        if entities.get("DISEASES"): print(f"🔹 DISEASES: {', '.join(entities['DISEASES'])}")
        if entities.get("MEDICATIONS_WITH_DETAILS"):
            meds = [f"{m['name']} ({m['dosage']})" for m in entities["MEDICATIONS_WITH_DETAILS"]]
            print(f"💊 MEDICATIONS: {', '.join(meds)}")
        
        # Step 7: Save Files
        transcript_path = save_transcript(final_transcript, audio_file)
        entity_path = ner_system.save_entities(entities, audio_file)
        summary_path = summarizer.save_summary(soap_note, patient_summary, audio_file)
        
        print(f"\n✅ PIPELINE COMPLETE!")
        print(f"📝 Transcript: {transcript_path}")
        print(f"📄 Summary (SOAP + Patient): {summary_path}")
        print(f"📊 Entities (JSON): {entity_path}")
        print(f"🧠 Knowledge Base: Updated in {VECTOR_DB_PATH}/")
        
    except Exception as e:
        print(f"💥 Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
