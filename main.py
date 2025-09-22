import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from faster_whisper import WhisperModel
import asyncio

model = WhisperModel("base", device="cpu")  

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🔗 Client connected")

    buffer = np.array([], dtype=np.float32)

    try:
        while True:
            data = await websocket.receive_bytes()
            audio_chunk = np.frombuffer(data, dtype=np.float32)

            buffer = np.concatenate([buffer, audio_chunk])

            segments, _ = model.transcribe(buffer, beam_size=5, language="en")

            text = " ".join([seg.text for seg in segments])
            await websocket.send_text(text)
            print("📝 Sent transcription:", text)

    except Exception as e:
        print("❌ Connection closed / Error:", e)


if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8001, reload=True)