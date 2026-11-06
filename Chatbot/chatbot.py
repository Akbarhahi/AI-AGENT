import os
import re
import sys
import gc
import json
import datetime
import wikipediaapi
import torch
import warnings
import pyttsx3
import speech_recognition as sr
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
from sentence_transformers import SentenceTransformer, util

warnings.filterwarnings("ignore")
logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ==========================================
# 1. LOCAL VOICE ENGINE (LAZY LOAD CPU STT)
# ==========================================

class LocalVoiceEngine:
    def __init__(self):
        # 1. TTS Engine (Windows SAPI5)
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 175)
        except Exception:
            self.tts_engine = None

        self.stt_model = None  # Dimuat nanti secara otomatis saat mic dipakai
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = 2.0
        self.recognizer.non_speaking_duration = 0.8
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True

    def _ensure_stt_loaded(self):
        """Memuat Whisper di CPU secara aman tanpa mengganggu VRAM GPU."""
        if self.stt_model is None:
            print("\n[VOICE] Menginisialisasi Whisper STT di CPU...")
            from faster_whisper import WhisperModel
            self.stt_model = WhisperModel("small", device="cpu", compute_type="int8")
            print("[VOICE] Whisper STT Siap Digunakan.")

    def speak(self, text: str):
        if not self.tts_engine:
            return
        if "```" in text or "Action: buat_script" in text:
            clean_text = "Saya sudah membuatkan dan menuliskan script yang diminta ke dalam file."
        else:
            clean_text = re.sub(r'[*_#`{}]', '', text)
        try:
            self.tts_engine.say(clean_text)
            self.tts_engine.runAndWait()
        except Exception:
            pass

    def listen(self) -> str:
        self._ensure_stt_loaded()
        with sr.Microphone() as source:
            print("\n[MIC] Mendengarkan... (Silakan bicara)")
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            try:
                audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=30)
            except sr.WaitTimeoutError:
                return ""

        temp_audio = "temp_voice.wav"
        with open(temp_audio, "wb") as f:
            f.write(audio.get_wav_data())

        try:
            segments, _ = self.stt_model.transcribe(
                temp_audio,
                language="id",
                beam_size=3
            )
            transcribed_text = " ".join([seg.text for seg in segments]).strip()
        except Exception as e:
            print(f"[STT Error]: {e}")
            transcribed_text = ""
        finally:
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
            
        return transcribed_text

voice = LocalVoiceEngine()


# ==========================================
# 2. VECTOR MEMORY & SESSION ENGINE
# ==========================================

class PersistentVectorMemory:
    def __init__(self, memory_file="agent_memory.pt", session_file="session_history.json", model_name="all-MiniLM-L6-v2"):
        self.memory_file = memory_file
        self.session_file = session_file
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.embedder = SentenceTransformer(model_name, device=self.device)
        self.memory_data = self._load_memory()

    def _load_memory(self):
        if os.path.exists(self.memory_file):
            try:
                return torch.load(self.memory_file, map_location=self.device, weights_only=False)
            except Exception:
                pass
        return {"texts": [], "embeddings": None}

    def auto_save_turn(self, user_msg: str, agent_msg: str):
        if len(user_msg.strip()) < 3 or user_msg.lower() in ["keluar", "exit"]:
            return
        entry = f"User: {user_msg} | Agent: {agent_msg}"
        new_embed = self.embedder.encode(entry, convert_to_tensor=True, device=self.device).unsqueeze(0)
        self.memory_data["texts"].append(entry)
        if self.memory_data["embeddings"] is None:
            self.memory_data["embeddings"] = new_embed
        else:
            self.memory_data["embeddings"] = torch.cat([self.memory_data["embeddings"], new_embed], dim=0)
        torch.save(self.memory_data, self.memory_file)

    def retrieve_context(self, query: str, top_k: int = 2, threshold: float = 0.35) -> str:
        if self.memory_data["embeddings"] is None or len(self.memory_data["texts"]) == 0:
            return ""
        query_embed = self.embedder.encode(query, convert_to_tensor=True, device=self.device)
        scores = util.cos_sim(query_embed, self.memory_data["embeddings"])[0]
        top_results = torch.topk(scores, k=min(top_k, len(scores)))
        memories = [self.memory_data['texts'][idx.item()] for score, idx in zip(top_results.values, top_results.indices) if score.item() >= threshold]
        return "\n".join(memories)

    def load_last_session(self, max_turns: int = 4) -> list:
        if os.path.exists(self.session_file):
            try:
                with open(self.session_file, "r", encoding="utf-8") as f:
                    return json.load(f)[-max_turns:]
            except Exception:
                pass
        return []

    def save_current_session(self, history: list):
        try:
            with open(self.session_file, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

memory_db = PersistentVectorMemory()


# ==========================================
# 3. TOOL REGISTRY
# ==========================================

wiki = wikipediaapi.Wikipedia(language='id', user_agent='SoftwareAgent/6.0 (student@example.com)')

def tool_wikipedia(topic: str) -> str:
    try:
        p = wiki.page(topic.strip().title())
        return p.summary.split('\n')[0] if p.exists() else f"Topik '{topic}' tidak ditemukan di Wikipedia."
    except Exception as e:
        return f"Error: {e}"

def tool_calculator(expression: str) -> str:
    expr = expression.lower().replace('x', '*').replace(':', '/')
    cleaned = re.sub(r'[^0-9+\-*/().]', '', expr)
    try:
        return str(eval(cleaned, {"__builtins__": None}, {})) if cleaned else "Format salah."
    except Exception:
        return "Gagal menghitung."

def tool_baca_file(nama_file: str) -> str:
    nama_file = nama_file.strip().strip("'\"")
    if not os.path.exists(nama_file):
        return f"File '{nama_file}' tidak ditemukan."
    try:
        with open(nama_file, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error: {e}"

def tool_list_files(_: str = "") -> str:
    try:
        return ", ".join([f for f in os.listdir('.') if os.path.isfile(f)])
    except Exception as e:
        return f"Error: {e}"

def tool_buat_script(argumen: str) -> str:
    try:
        if "|||" not in argumen:
            return "Gagal: Format harus 'nama_file.ext|||isi_kode'"
        nama_file, kode = argumen.split("|||", 1)
        nama_file = nama_file.strip().strip("'\"")
        with open(nama_file, "w", encoding="utf-8") as f:
            f.write(kode.strip())
        return f"File '{nama_file}' berhasil dibuat dan disimpan."
    except Exception as e:
        return f"Gagal membuat script: {e}"

def tool_current_time(_: str = "") -> str:
    return datetime.datetime.now().strftime('%A, %d-%m-%Y pukul %H:%M:%S')

def tool_optimize_system(_: str = "") -> str:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        return f"VRAM GPU dibersihkan. Sisa terpakai: {torch.cuda.memory_allocated() / (1024**2):.2f} MB"
    return "Optimasi selesai."

AVAILABLE_TOOLS = {
    "wikipedia": tool_wikipedia,
    "calculator": tool_calculator,
    "baca_file": tool_baca_file,
    "list_files": tool_list_files,
    "buat_script": tool_buat_script,
    "current_time": tool_current_time,
    "optimize": tool_optimize_system
}


# ==========================================
# 4. REASONING ENGINE (QWEN 3B DI CUDA)
# ==========================================

class AutonomousVoiceAgent:
    def __init__(self, model_id="Qwen/Qwen2.5-3B-Instruct"):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Memuat Model LLM ({model_id})...")
        print(f"[INFO] Komputasi LLM: {self.device.upper()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        ).to(self.device)

        self.history = memory_db.load_last_session()
        self.system_prompt = (
            "Kamu adalah asisten AI software engineering dan coding berbahasa Indonesia yang cerdas, tepat sintaks, dan ringkas.\n"
            "ATURAN BAHASA MUTLAK: WAJIB MENJAWAB DALAM BAHASA INDONESIA.\n"
            "Daftar Tools:\n"
            "- buat_script[nama_file.ext|||isi_kode]: Menulis dan menyimpan skrip C#, C++, atau Python ke file fisik.\n"
            "- baca_file[nama_file]: Membaca isi file teks/kode.\n"
            "- list_files[]: Melihat daftar file di folder.\n"
            "- wikipedia[topik]: Ensiklopedia fakta.\n"
            "- calculator[ekspresi]: Operasi matematika murni.\n"
            "- current_time[]: Informasi waktu saat ini.\n"
            "- optimize[]: Membersihkan cache VRAM GPU.\n\n"
            "Aturan Penulisan Kode:\n"
            "1. C#: Pastikan mencakup 'using System;', class, method, dan sintaks yang valid.\n"
            "2. C++: Wajib menyertakan '#include <iostream>', int main(), dan namespace/std:: yang tepat.\n"
            "3. Python: Gunakan format sintaks bersih, penanganan error, dan indentasi yang benar.\n"
            "4. Jika diminta membuat/menyimpan file, panggil Action: buat_script[nama_file.ext|||isi_kode]\n"
            "5. Jika hanya ingin melihat kode di layar, tampilkan langsung di respon tanpa tool."
        )

    def _generate_from_messages(self, messages: list, max_tokens=450) -> str:
        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([prompt_text], return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                repetition_penalty=1.12,
                pad_token_id=self.tokenizer.eos_token_id
            )
        generated_ids = outputs[0][len(inputs.input_ids[0]):]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def get_response(self, user_query: str) -> str:
        if len(self.history) > 4:
            self.history = self.history[-3:]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        recalled = memory_db.retrieve_context(user_query)
        sys_prompt = self.system_prompt
        if recalled:
            sys_prompt += f"\n\n[MEMORI RELEVAN]:\n{recalled}"

        messages = [{"role": "system", "content": sys_prompt}]
        for turn in self.history:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["agent"]})
        messages.append({"role": "user", "content": user_query})

        raw_gen = self._generate_from_messages(messages, max_tokens=120)
        action_match = re.search(r"Action:\s*([a-zA-Z_]+)(?:\[([\s\S]*?)\])?$", raw_gen, re.IGNORECASE)

        if action_match:
            tool_name = action_match.group(1).lower().strip()
            tool_param = action_match.group(2) if action_match.group(2) else ""
            
            if tool_name in AVAILABLE_TOOLS:
                obs = AVAILABLE_TOOLS[tool_name](tool_param)
                messages.append({"role": "assistant", "content": f"Action: {tool_name}[{tool_param}]\nObservation: {obs}\nJawablah user:"})
                raw_answer = self._generate_from_messages(messages, max_tokens=450)
            else:
                raw_answer = raw_gen
        else:
            raw_answer = self._generate_from_messages(messages, max_tokens=450)

        clean_answer = raw_answer.split("Thought:")[0].split("Action:")[0].replace("Final Answer:", "").strip()
        if not clean_answer:
            clean_answer = raw_answer.strip()

        print(f"Agent: {clean_answer}\n")
        voice.speak(clean_answer)

        self.history.append({"user": user_query, "agent": clean_answer})
        memory_db.save_current_session(self.history)
        memory_db.auto_save_turn(user_query, clean_answer)
        return clean_answer


# ==========================================
# 5. RUNNER
# ==========================================
if __name__ == "__main__":
    agent = AutonomousVoiceAgent()
    print("\n--- Autonomous AI Software Agent Siap ---")
    print("Ketik teks biasa atau tekan ENTER kosong untuk bicara via MIC.")
    print("Ketik 'keluar' untuk mengakhiri.\n")

    while True:
        try:
            msg = input("You (Ketik atau tekan Enter untuk Mic): ").strip()
            
            if not msg:
                msg = voice.listen()
                if not msg:
                    print("[MIC] Tidak ada suara terdeteksi.")
                    continue
                print(f"You (Voice Recognized): {msg}")

            if msg.lower() in ["keluar", "exit"]:
                print("Agent: Sampai jumpa!")
                voice.speak("Sampai jumpa!")
                break
            
            agent.get_response(msg)
        except KeyboardInterrupt:
            break