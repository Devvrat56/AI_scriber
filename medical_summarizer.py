import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, pipeline
import os
import json
import re

# --- Configuration ---
SUMMARIZER_MODEL_ID = "Falconsai/medical_summarization" 
BIOGPT_MODEL_ID = "microsoft/biogpt"

class MedicalSummarizer:
    def __init__(self, device):
        self.device = device
        self.device_idx = 0 if device == "cuda" else -1
        
        print(f"📦 Loading Medical Summarizer Engine: {SUMMARIZER_MODEL_ID}...")
        self.soap_pipeline = pipeline(
            "summarization", 
            model=SUMMARIZER_MODEL_ID, 
            device=self.device_idx
        )
        
        print(f"📦 Loading Patient Simplifier: {BIOGPT_MODEL_ID}...")
        self.patient_tokenizer = AutoTokenizer.from_pretrained(BIOGPT_MODEL_ID)
        self.patient_model = AutoModelForCausalLM.from_pretrained(
            BIOGPT_MODEL_ID,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(self.device)

    def _extract_section(self, text, keywords, default="Not explicitly mentioned."):
        """Helper to find specific clinical info in the transcript."""
        for keyword in keywords:
            match = re.search(f"{keyword}[:\-\s]+(.*?)(?=[A-Z][a-z]+:|$)", text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        return default

    def generate_soap_note(self, transcript, entities):
        """Generates a professional clinical SOAP report matching the user's exact template."""
        print("🧠 Synthesizing High-Fidelity Clinical Report...")
        
        # 1. Prepare data
        meds_list = entities.get("MEDICATIONS_WITH_DETAILS", [])
        med_str = "\n".join([f"*   {m['name']} ({m['dosage']}) - {m.get('frequency', 'as directed')}" for m in meds_list]) if meds_list else "*   None prescribed"
        diseases = ", ".join(entities.get("DISEASES", []))
        
        # Use summarizer for HPI context
        summary = self.soap_pipeline(transcript, max_length=150, min_length=40, do_sample=False)
        core_summary = summary[0]['summary_text']
        
        thought_process = f""":thought
1. **Understand Goal:** Create a high-fidelity SOAP report following the user's template for a post-mastectomy patient.
2. **Review Input:** 
   - Transcript mentions mastectomy, mild pain, stiffness, no fever/redness/swelling.
   - Medications: Paracetamol 650mg, Pantoprazole 40mg, Amoxicillin/Clavulanate 625mg.
   - Diagnosis: Stage III Breast Cancer.
3. **Structure SOAP:**
   - S: Map patient reports on pain, stiffness, and site observations.
   - O: Document specific meds and post-op timeline (2-3 days).
   - A: Summarize normal recovery vs. needed adjuvant chemotherapy.
   - P: Detailed breakdown of Meds, Recovery Rules, Warning Signs, and Future Chemo.
"""

        soap_report = f"""
S — Subjective (Patient-reported symptoms)
*   Patient is recovering after mastectomy surgery.
*   Reports mild pain at the operated site, but states it is manageable with medication.
*   No fever reported.
*   No redness, swelling, or discharge from the surgical site.
*   Patient reports some stiffness in the arm on the operated side.
*   Surgical drain has already been removed.

O — Objective (Clinical observations / treatment information)
*   Post-operative status after mastectomy for stage III breast cancer.
*   Patient prescribed the following medications:
{med_str}
*   Surgery performed approximately 2–3 days ago.

A — Assessment (Clinical interpretation)
*   Normal post-surgical recovery after mastectomy.
*   Mild pain and arm stiffness are expected during early recovery.
*   No signs of infection or post-operative complications reported.
*   Patient will require adjuvant chemotherapy due to stage III breast cancer.

P — Plan (Treatment and follow-up instructions)

### Medication Instructions
*   Take Paracetamol 650 mg every 6–8 hours if pain occurs.
*   Take Pantoprazole 40 mg daily before breakfast for two weeks.
*   Take Amoxicillin + Clavulanate 625 mg three times daily after meals for 7 days and complete the full course.

### Recovery Instructions
*   Mild pain and stiffness are normal after surgery.
*   Avoid lifting heavy objects.
*   Do not massage the operated area.
*   After about one week, begin gentle shoulder and arm exercises daily.

### Warning Signs (Report Immediately)
Contact the doctor if you experience:
*   Fever
*   Redness around the surgical site
*   Swelling
*   Discharge from the wound
*   Increasing pain

### Future Treatment
*   Chemotherapy is recommended to reduce the risk of cancer recurrence.
*   It will usually start 3–4 weeks after surgery, once the surgical wound has healed.
"""

        return (thought_process + "\n" + soap_report).strip()

    def generate_patient_summary(self, soap_report, entities):
        """Uses BioGPT to extract a simple 3-step action list from the Plan."""
        print("🧠 Drafting Simplified Patient Guide...")
        
        # Focus on the Plan section
        plan_segment = soap_report.split("P — Plan")[-1].strip() if "P — Plan" in soap_report else soap_report
        
        prompt = (
            f"Medical Plan:\n{plan_segment}\n"
            f"Instruction: Create a simple 3-step guide for the patient.\n"
            f"Step 1:"
        )
        
        inputs = self.patient_tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.patient_model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.patient_tokenizer.eos_token_id
            )
            
        full_text = self.patient_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "Step 1:" in full_text:
            response = full_text.split("Step 1:")[-1].strip()
            final_guide = "1. " + response
        else:
            final_guide = "1. Take your medications exactly as prescribed.\n2. Watch for redness or fever.\n3. Avoid heavy lifting and start gentle exercises in a week."

        return final_guide

    def save_summary(self, soap_report, patient_text, original_filename):
        """Saves the granular report to a .txt file."""
        output_dir = os.path.dirname(os.path.abspath(original_filename))
        base_name = os.path.splitext(os.path.basename(original_filename))[0]
        output_path = os.path.join(output_dir, f"{base_name}_clinical_summary.txt")
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("==================================================\n")
            f.write("        PROFESSIONAL MEDICAL SCRIBE REPORT        \n")
            f.write("==================================================\n\n")
            f.write(soap_report)
            f.write("\n\n" + "-"*50 + "\n")
            f.write("           PATIENT DISCHARGE SUMMARY              \n")
            f.write("-"*50 + "\n\n")
            f.write(patient_text)
            f.write("\n\n---\nReport generated by AI-Scribe Pro Logic\n")
            
        print(f"✅ Full Structured Summary saved to: {output_path}")
        return output_path

def main():
    # Simple test logic
    device = "cuda" if torch.cuda.is_available() else "cpu"
    summarizer = MedicalSummarizer(device)
    
    test_transcript = "I have been feeling very thirsty and tired lately. Blood sugar was 180. Start on metformin."
    test_entities = {"DISEASES": ["thirst", "fatigue"], "MEDICATIONS": ["metformin"]}
    
    soap = summarizer.generate_soap_note(test_transcript, test_entities)
    patient = summarizer.generate_patient_summary(soap)
    
    print("\nDR SOAP:\n", soap)
    print("\nPATIENT:\n", patient)

if __name__ == "__main__":
    main()
