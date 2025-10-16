# 🎧 StreamTranscriber  
*A real-time audio transcription proof-of-concept using Faster-Whisper and WebSockets*

---

## 🚀 Overview
Conventional transcription systems send long recordings (sometimes 90 s or more) for processing and make users wait tens of seconds for results.  
This PoC prototype demonstrates how to **stream audio continuously** to a backend and receive **incremental transcriptions in near real-time**.

It was created as a small project for a German AI startup exploring the move from slow batch uploads to a fully streamed pipeline.

---

## 🧠 What It Does
- Streams raw PCM `float32` audio chunks from client → FastAPI backend through WebSockets  
- Backend transcribes chunks on the fly with **[Faster-Whisper](https://github.com/guillaumekln/faster-whisper)**  
- Sends partial transcription text back to the client after each chunk  
- Gives the user the feeling of *live captioning* instead of waiting for large batch results

---

## 🏗️ Tech Stack

| Layer | Technology |
|-------|-------------|
| **Backend Framework** | FastAPI + WebSockets |
| **Speech-to-Text Model** | Faster-Whisper (`base`) |
| **Language** | Python 3.10+ |
| **Client** | Python WebSockets sender (replaceable with microphone input) |
| **Audio Format** | Mono 16 kHz `float32` PCM |

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/rohan9024/StreamTranscriber.git
cd StreamTranscriber
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
# Activate it
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

### 4. Run the Server
```bash
uvicorn server:app --host 127.0.0.1 --port 8001 --reload
```

### 5. Send Audio from the Client
```bash
python client.py
```

**Note:**
- Your `sample_16k.wav` file should be mono and 16 kHz.
- If your file is stereo, the client automatically converts it to mono.

---

## 📊 How It Works
1. Client loads a `.wav` file, breaks it into ~2 s chunks, and sends each chunk via WebSocket.
2. Server receives chunk → converts from bytes to numpy array → transcribes with Faster-Whisper.
3. Server immediately returns partial text to the client.
4. The process continues until the entire audio stream is processed.

---

## 🧩 Example Output
Console snippet while running:

```text
📦 Sent 32000 samples
📝 Partial transcription: Hello everyone and welcome to the meeting...
📦 Sent 32000 samples
📝 Partial transcription: Today we'll discuss progress on the new release...
```

---

## 🚀 Results
| Approach | Delay Before Text | User Experience |
|----------|-------------------|-----------------|
| 90 s file upload | ≈ 45 s | Feels batch-processed |
| Streamed PCM chunks | 1 – 2 s | Feels live and conversational |

By streaming smaller chunks, the user starts seeing text updates almost instantly instead of waiting for the full recording to complete.

---

## 🛠️ Possible Extensions
- Real-time microphone input (continuous streaming)
- Better buffering / overlap for smoother context
- Frontend dashboard that displays text live (React / JS)
- Integration with summarization or meeting-notes generators

---

## 💡 Motivation
This PoC came from a common startup bottleneck:

> "Recording 90 s of audio and waiting another 45 s for a transcript isn't practical for live use."

**StreamTranscriber** proves that even lightweight infrastructure can deliver real-time AI transcription using open-source tools.

---

## 📦 Requirements
Create a file `requirements.txt` (if you don't have one already) with:

```text
fastapi==0.109.0
uvicorn==0.25.0
numpy==1.26.0
soundfile==0.12.1
websockets==12.0
faster-whisper==0.9.0
```

---

## 🤝 Acknowledgements
- **Faster-Whisper** for efficient ASR inference
- **FastAPI** for its excellent async & WebSocket support
- Open-source community for continued innovation in AI infrastructure

---

## 🧑‍💻 Author
**Rohan R. Wandre**  
ML Engineer


---

Feel free to fork, modify, or use this project as a reference for building real-time transcription pipelines.
