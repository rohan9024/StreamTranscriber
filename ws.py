import numpy as np
from fastapi import WebSocket
from app.transcriber import transcribe_chunk

async def audio_stream_handler(websocket: WebSocket):
    await websocket.accept()
    print("🔗 Client connected")

    audio_buffer = np.array([], dtype=np.float32)

    try:
        while True:
            data = await websocket.receive_bytes()
            audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            audio_buffer = np.concatenate([audio_buffer, audio_chunk])

            if len(audio_buffer) >= 16000:
                text = transcribe_chunk(audio_buffer[:16000])  
                await websocket.send_text(text)
                audio_buffer = audio_buffer[16000:]  

    except Exception as e:
        print("❌ Connection closed:", e)
        await websocket.close()
