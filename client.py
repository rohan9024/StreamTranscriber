import asyncio
import websockets
import soundfile as sf
import numpy as np


async def send_audio(file_path):
    uri = "ws://127.0.0.1:8001/ws/audio"
    async with websockets.connect(uri) as websocket:
        print("✅ Connected to server")

        audio, sr = sf.read(file_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  

        chunk_size = sr * 2

        for start in range(0, len(audio), chunk_size):
            chunk = audio[start:start + chunk_size]
            await websocket.send(chunk.tobytes())
            print(f"📦 Sent {len(chunk)} samples")

            text = await websocket.recv()
            print("📝 Partial transcription:", text)

        print("✅ Finished sending audio")


if __name__ == "__main__":
    asyncio.run(send_audio("sample_16k.wav"))