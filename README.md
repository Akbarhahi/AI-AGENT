## 🚀 Key Features / Fitur Utama

- **🧠 Autonomous ReAct Reasoning Core**  
  Menggunakan model lokal **Qwen2.5-3B-Instruct** yang berjalan di GPU dengan siklus *Thought $\rightarrow$ Action $\rightarrow$ Observation $\rightarrow$ Final Answer* untuk memecahkan masalah dan mengeksekusi perintah secara mandiri.

- **👁️ Real-time OCR & Screen Debugging**  
  Terintegrasi dengan **EasyOCR** untuk membaca pesan *error*, *stack trace*, dan teks visual langsung dari seluruh layar atau jendela aplikasi aktif (seperti Unity Editor, VS Code, atau browser) tanpa membebankan VRAM.

- **💾 Persistent Vector Memory & Auto-Ingestion**  
  Sistem memori jangka panjang berbasis **Sentence-Transformers (`all-MiniLM-L6-v2`)** yang secara otomatis mengindeks dan menyimpan riwayat percakapan secara semantik ke dalam basis data vektor lokal (`agent_memory.pt`).

- **🎙️ Voice-Enabled I/O (Offline STT & TTS)**  
  Mendukung interaksi suara penuh secara lisan:
  - **Speech-to-Text (STT):** Transkripsi suara *real-time* berbasis **Faster-Whisper (Small)** yang dioptimalkan di CPU.
  - **Text-to-Speech (TTS):** Respon balik suara alami menggunakan **pyttsx3**.

- **🛠️ Automated Software Engineering & Script Tools**  
  Dipelengkapi dengan *Tool Registry* bawaan untuk tugas-tugas pemrogramm:
  - `buat_script`: Menulis dan menyimpan skrip C#, C++, atau Python secara otomatis.
  - `baca_file` & `list_files`: Membaca dan menjelajahi struktur berkas lokal.
  - `calculator` & `current_time`: Utilitas perhitungan spasial/matematika dan waktu.

- **⚡ Optimized Hybrid Hardware Offloading (Low VRAM Friendly)**  
  Dirancang khusus untuk berjalan stabil pada GPU laptop dengan VRAM terbatas (NVIDIA RTX 3060 6GB):
  - Model LLM dimuat menggunakan **4-bit NF4 Quantization (BitsAndBytes)** (~2.2 GB VRAM).
  - Modul OCR dan STT dialihkan penuh ke **CPU & RAM**, mencegah bahaya *CUDA Out-Of-Memory (OOM)* saat membuka game engine/IDE bersamaan.
  - Fitur `optimize` bawaan untuk membersihkan *cache memory* GPU secara dinamis.