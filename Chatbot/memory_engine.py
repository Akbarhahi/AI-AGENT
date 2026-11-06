import os
import torch
from sentence_transformers import SentenceTransformer, util

class PersistentVectorMemory:
    def __init__(self, memory_file="agent_memory.pt", model_name="all-MiniLM-L6-v2"):
        self.memory_file = memory_file
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.embedder = SentenceTransformer(model_name, device=self.device)
        self.memory_data = self._load_memory()

    def _load_memory(self):
        if os.path.exists(self.memory_file):
            try:
                data = torch.load(self.memory_file, map_location=self.device)
                print(f"[MEMORY] Berhasil memuat {len(data['texts'])} memori tersimpan.")
                return data
            except Exception:
                pass
        return {"texts": [], "embeddings": None}

    def save_memory(self, text: str):
        """Menyimpan fakta baru ke memori permanen."""
        text = text.strip()
        if not text:
            return "Teks kosong, tidak dapat disimpan."
        
        # Buat vektor embedding
        new_embed = self.embedder.encode(text, convert_to_tensor=True, device=self.device).unsqueeze(0)
        
        self.memory_data["texts"].append(text)
        if self.memory_data["embeddings"] is None:
            self.memory_data["embeddings"] = new_embed
        else:
            self.memory_data["embeddings"] = torch.cat([self.memory_data["embeddings"], new_embed], dim=0)

        # Simpan ke file disk
        torch.save(self.memory_data, self.memory_file)
        return f"Memori disimpan: '{text}'"

    def retrieve_memory(self, query: str, top_k: int = 2, threshold: float = 0.35) -> str:
        """Mencari memori paling relevan berdasarkan Cosine Similarity."""
        if self.memory_data["embeddings"] is None or len(self.memory_data["texts"]) == 0:
            return "Belum ada memori yang tersimpan."

        query_embed = self.embedder.encode(query, convert_to_tensor=True, device=self.device)
        scores = util.cos_sim(query_embed, self.memory_data["embeddings"])[0]

        top_results = torch.topk(scores, k=min(top_k, len(scores)))
        relevant_memories = []

        for score, idx in zip(top_results.values, top_results.indices):
            if score.item() >= threshold:
                relevant_memories.append(f"- {self.memory_data['texts'][idx.item()]} (Skor: {score.item():.2f})")

        if not relevant_memories:
            return "Tidak ada memori lama yang cocok dengan query ini."
        
        return "\n".join(relevant_memories)