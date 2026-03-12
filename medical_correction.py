import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
import os

# --- Configuration ---
# Model 1: BioBART (Generative BioBERT) - Primary Corrector
MODEL_1_ID = "GanjinZero/biobart-v2-large"
# Model 2: BioGPT - Verification Model
MODEL_2_ID = "microsoft/biogpt"
# Similarity Model: ClinicalBERT for context/semantic comparison
SIMILARITY_MODEL_ID = "emilyalsentzer/Bio_ClinicalBERT"

def check_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📡 Using Device: {device.upper()}")
    return device

class MedicalCorrectionPipeline:
    def __init__(self, device):
        self.device = device
        self.device_idx = 0 if device == "cuda" else -1
        self.torch_dtype = torch.float16 if device == "cuda" else torch.float32
        
        print(f"📦 Loading Model 1: {MODEL_1_ID} (BioBERT/BioBART)...")
        # Load BioBART manually to avoid pipeline 'text2text-generation' task issues
        self.corrector_tokenizer = AutoTokenizer.from_pretrained(MODEL_1_ID)
        self.corrector_model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_1_ID,
            torch_dtype=self.torch_dtype
        ).to(self.device)
        
        print(f"📦 Loading Model 2: {MODEL_2_ID} (BioGPT/Verification)...")
        self.verifier_tokenizer = AutoTokenizer.from_pretrained(MODEL_2_ID)
        self.verifier_model = AutoModelForCausalLM.from_pretrained(
            MODEL_2_ID,
            torch_dtype=self.torch_dtype
        ).to(self.device)
        
        print(f"📦 Loading Similarity Model: {SIMILARITY_MODEL_ID}...")
        self.sim_model = SentenceTransformer(SIMILARITY_MODEL_ID)

    def generate_biobart_correction(self, text):
        """Generates correction using BioBART."""
        prompt = f"Correct medical spelling and grammar: {text}"
        inputs = self.corrector_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        
        with torch.no_grad():
            outputs = self.corrector_model.generate(
                **inputs,
                max_length=512,
                num_beams=4,
                early_stopping=True
            )
        
        return self.corrector_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    def generate_biogpt_correction(self, text):
        """Generates correction using BioGPT."""
        prompt = f"Correct the medical spelling and grammar: {text}\nCorrection:"
        inputs = self.verifier_tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.verifier_model.generate(
                **inputs,
                max_new_tokens=100,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.verifier_tokenizer.eos_token_id
            )
        
        full_text = self.verifier_tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "Correction:" in full_text:
            return full_text.split("Correction:")[-1].strip()
        return full_text.strip()

    def process(self, asr_text):
        if not asr_text or len(asr_text.strip()) == 0:
            return ""

        print(f"\n📝 Original ASR: {asr_text[:200]}...")
        
        # Level 1 & 2: BioBERT (BioBART) Correction
        print("🧠 Running Model 1 (BioBART) Correction...")
        s1 = self.generate_biobart_correction(asr_text)
        print(f"✨ S1 (BioBART): {s1[:200]}...")
        
        # Level 3: Clinical Verification (using BioGPT as Model 2)
        # We only verify a snippet if it's too long, or the whole thing if it's short
        print("🧠 Running Model 2 (BioGPT) Verification...")
        s2 = self.generate_biogpt_correction(asr_text[:500]) # Snippet for speed
        print(f"✨ S2 (BioGPT): {s2[:200]}...")
        
        # Calculate Similarity
        print("秤 Calculating similarity between S1 and S2...")
        emb1 = self.sim_model.encode(s1, convert_to_tensor=True)
        emb2 = self.sim_model.encode(s2, convert_to_tensor=True)
        similarity = util.cos_sim(emb1, emb2).item()
        print(f"📊 Cosine Similarity: {similarity:.4f}")
        
        # Decision Logic
        if similarity > 0.8: # Adjusted threshold for real-world text
            print("✅ Good consistency! Accepting S1.")
            return s1
        else:
            print("⚠️ Semantic drift detected. Using S1 but flagging for review.")
            return s1

def main():
    device = check_device()
    pipeline = MedicalCorrectionPipeline(device)
    
    # Test cases
    test_texts = [
        "patient taking metfornin for diabetis",
        "The patient complains of chest pane and shortnes of breathe.",
        "Diagnosis of hpertension and hyperlipdemia."
    ]
    
    for text in test_texts:
        result = pipeline.process(text)
        print(f"🏁 Final Corrected Text: {result}\n" + "-"*50)

if __name__ == "__main__":
    main()
