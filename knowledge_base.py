import os
import json
import faiss
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
import pickle
import re

# --- Configuration ---
VECTOR_DB_PATH = "medical_knowledge_base"
EMBEDDING_MODEL_ID = "emilyalsentzer/Bio_ClinicalBERT"

class MedicalKnowledgeBase:
    def __init__(self, device="cpu"):
        self.device = device
        print(f"📦 Initializing Advanced Knowledge Base with {EMBEDDING_MODEL_ID} on {device.upper()}...")
        
        # 2. Use GPU for the Embedding Model (Fix)
        self.model = SentenceTransformer(EMBEDDING_MODEL_ID, device=self.device)
        
        self.db_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), VECTOR_DB_PATH)
        if not os.path.exists(self.db_dir):
            os.makedirs(self.db_dir)
            
        self.index_path = os.path.join(self.db_dir, "faiss_index.bin")
        self.metadata_path = os.path.join(self.db_dir, "metadata.pkl")
        
        self.dimension = 768 # Bio_ClinicalBERT dimension
        
        # 1 & 6. Use Cosine Similarity (IndexFlatIP) and HNSW for scalability
        # IndexHNSWFlat is much faster for large datasets O(log N)
        self.index = faiss.IndexHNSWFlat(self.dimension, 32, faiss.METRIC_INNER_PRODUCT)
        self.metadata = []
        
        # Load existing index if available
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            print("🔄 Loading existing Knowledge Base...")
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            print(f"✅ Loaded {len(self.metadata)} medical records.")

    def chunk_transcript(self, text, max_chars=600):
        """3. Smart Chunking Strategy: Splits by sentence to avoid breaking medical context."""
        # Simple sentence splitter using regex
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < max_chars:
                current_chunk += sentence + " "
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def add_consultation(self, transcript, entities, audio_filename):
        """
        Adds a new consultation record.
        4. Store Structured Medical Entities Separately in metadata.
        """
        timestamp = datetime.now().isoformat()
        consultation_id = f"CONS_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        chunks = self.chunk_transcript(transcript)
        
        # 1. Normalize embeddings for Cosine Similarity
        embeddings = self.model.encode(chunks, normalize_embeddings=True)
        
        # Add to FAISS
        self.index.add(np.array(embeddings).astype('float32'))
        
        # Store enriched metadata
        for chunk in chunks:
            self.metadata.append({
                "consultation_id": consultation_id,
                "text": chunk,
                "timestamp": timestamp,
                "audio_source": audio_filename,
                # 4. Structured entity storage
                "disease_entities": entities.get("DISEASES", []),
                "medication_entities": entities.get("MEDICATIONS", []),
                "detailed_meds": entities.get("MEDICATIONS_WITH_DETAILS", []),
                "unlinked_dosages": entities.get("UNLINKED_DOSAGES", [])
            })
            
        # Persistence
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
            
        print(f"💾 Ingested {len(chunks)} contextual chunks into the Knowledge Base.")

    def search(self, query, top_k=5, disease_filter=None):
        """
        5. Combined Text Search + Entity Filtering.
        """
        if self.index.ntotal == 0:
            return []
            
        # 1. Vector similarity search with normalization
        query_vector = self.model.encode([query], normalize_embeddings=True)
        distances, indices = self.index.search(np.array(query_vector).astype('float32'), top_k)
        
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx == -1: continue
            
            record = self.metadata[idx]
            similarity_score = float(distances[0][i])
            
            # 5. Entity-based filtering
            if disease_filter:
                if disease_filter.lower() not in [d.lower() for d in record["disease_entities"]]:
                    continue
            
            record["similarity"] = similarity_score
            results.append(record)
            
        return results

def main():
    # Test script with the new advanced architecture
    device = "cuda" if faiss.get_num_gpus() > 0 else "cpu"
    kb = MedicalKnowledgeBase(device=device)
    
    test_transcript = "The patient was diagnosed with breast cancer. We prescribed paracetamol 650 mg and amoxicillin 500 mg. Recovery is going well."
    test_entities = {
        "DISEASES": ["breast cancer"], 
        "MEDICATIONS": ["paracetamol", "amoxicillin"],
        "MEDICATIONS_WITH_DETAILS": [
            {"name": "paracetamol", "dosage": "650 mg"},
            {"name": "amoxicillin", "dosage": "500 mg"}
        ]
    }
    
    kb.add_consultation(test_transcript, test_entities, "clinical_session_01.mp4")
    
    print("\n🔍 Advanced Search: 'cancer treatment' with disease filter...")
    results = kb.search("cancer treatment", disease_filter="breast cancer")
    for r in results:
        print(f"[{r['similarity']:.3f}] {r['text']}")
        print(f"   💊 Meds: {', '.join([m['name'] for m in r['detailed_meds']])}")

if __name__ == "__main__":
    main()
