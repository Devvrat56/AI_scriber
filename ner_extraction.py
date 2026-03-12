import spacy
import scispacy
from spacy.pipeline import EntityRuler
import re
import json
import os

# --- Configuration ---
NER_MODEL_ID = "en_ner_bc5cdr_md" 

class MedicalNERPipeline:
    def __init__(self):
        print(f"📦 Loading SciSpacy NER Model: {NER_MODEL_ID}...")
        try:
            self.nlp = spacy.load(NER_MODEL_ID)
            
            # Add Dosage and Frequency Patterns using EntityRuler
            # This helps catch "500mg", "650 mg", "twice a day", etc.
            ruler = self.nlp.add_pipe("entity_ruler", before="ner")
            patterns = [
                # Dosages (mg, ml, mcg, g)
                {"label": "DOSAGE", "pattern": [{"IS_DIGIT": True}, {"LOWER": {"IN": ["mg", "ml", "g", "mcg", "ug", "units", "tablets", "tablet", "pill", "pills"]}}]},
                {"label": "DOSAGE", "pattern": [{"LIKE_NUM": True}, {"LOWER": {"IN": ["mg", "ml", "g", "mcg", "ug", "units", "tablets", "tablet", "pill", "pills"]}}]},
                {"label": "DOSAGE", "pattern": [{"TEXT": {"REGEX": r"^\d+(mg|ml|g|mcg|units)$"}}]},
                
                # Frequencies
                {"label": "FREQUENCY", "pattern": [{"LOWER": "twice"}, {"LOWER": "a"}, {"LOWER": "day"}]},
                {"label": "FREQUENCY", "pattern": [{"LOWER": "once"}, {"LOWER": "a"}, {"LOWER": "day"}]},
                {"label": "FREQUENCY", "pattern": [{"LOWER": "every"}, {"IS_DIGIT": True}, {"LOWER": "hours"}]},
                {"label": "FREQUENCY", "pattern": [{"LOWER": "daily"}]},
                {"label": "FREQUENCY", "pattern": [{"LOWER": "bid"}]}, # Common medical shorthand
                {"label": "FREQUENCY", "pattern": [{"LOWER": "tid"}]},
                {"label": "FREQUENCY", "pattern": [{"LOWER": "qid"}]},
            ]
            ruler.add_patterns(patterns)
            
        except Exception as e:
            print(f"❌ Error loading SciSpacy model or rules: {e}")
            raise e

    def extract_entities(self, text):
        """Extracts medical entities and pairs medications with dosages."""
        print("🧠 Extracting detailed medical entities...")
        doc = self.nlp(text)
        
        results = {
            "DISEASES": [],
            "MEDICATIONS": [], # Raw list of chemicals/drugs
            "MEDICATIONS_WITH_DETAILS": [], # Linked Medication + Dosage
            "OTHER": []
        }
        
        # 1. Collect all distinct entities
        found_chemicals = []
        found_dosages = []
        
        for ent in doc.ents:
            label = ent.label_
            text_val = ent.text.strip()
            
            if label == "DISEASE":
                if text_val not in results["DISEASES"]:
                    results["DISEASES"].append(text_val)
            elif label == "CHEMICAL":
                found_chemicals.append(ent)
                if text_val not in results["MEDICATIONS"]:
                    results["MEDICATIONS"].append(text_val)
            elif label == "DOSAGE":
                found_dosages.append(ent)
            else:
                if text_val not in results["OTHER"]:
                    results["OTHER"].append(text_val)
        
        # 2. Link Medications with Dosages (Smart Linking)
        # We look for Dosages that appear close to Chemicals (within 5 tokens)
        used_dosages = set()
        
        for chem in found_chemicals:
            linked_detail = {"name": chem.text, "dosage": "Not specified", "frequency": "Not specified"}
            
            # Search for nearby dosage
            for dose in found_dosages:
                # Check distance (if they are within 10 tokens of each other)
                if abs(dose.start - chem.end) < 8 or abs(chem.start - dose.end) < 8:
                    linked_detail["dosage"] = dose.text
                    used_dosages.add(dose)
                    break # Assign the first closest dosage
            
            results["MEDICATIONS_WITH_DETAILS"].append(linked_detail)
            
        # Add a section for unlinked dosages (e.g. "take 500mg" when drug name wasn't clear)
        results["UNLINKED_DOSAGES"] = [d.text for d in found_dosages if d not in used_dosages]
        
        return results

    def save_entities(self, entities, original_filename):
        """Saves extracted entities to a JSON file."""
        output_dir = os.path.dirname(os.path.abspath(original_filename))
        base_name = os.path.splitext(os.path.basename(original_filename))[0]
        output_path = os.path.join(output_dir, f"{base_name}_medical_details.json")
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(entities, f, indent=4)
            
        print(f"✅ Medical details saved to: {output_path}")
        return output_path

def main():
    # Enhanced test
    test_text = "The patient is prescribed paracetamol 650 mg twice a day for pain and amoxicillin 500mg for infection. Also follow up on diabetes."
    try:
        ner = MedicalNERPipeline()
        entities = ner.extract_entities(test_text)
        print("\n--- EXTRACTED MEDICAL DETAILS ---")
        print(json.dumps(entities, indent=2))
    except Exception as e:
        print(f"💥 Testing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
