import asyncio
import websockets
import soundfile as sf
import numpy as np

SAMPLE_RATE = 16000
WINDOW_SECONDS = 2.0   
CHUNK_SIZE = int(SAMPLE_RATE * WINDOW_SECONDS)

async def sender(ws, audio):
    for start in range(0, len(audio), CHUNK_SIZE):
        chunk = audio[start:start + CHUNK_SIZE]
        await ws.send(chunk.tobytes())
        print(f"📦 Sent {len(chunk)} samples")
    await ws.send("EOS")
    print("📨 Sent EOS")

async def receiver(ws):
    try:
        async for msg in ws:
            if msg == "DONE":
                print("✅ Server DONE, closing")
                break
            print("📝 Commit:", msg)
    except websockets.exceptions.ConnectionClosed:
        print("🔚 Server closed the connection")

async def run(file_path):
    uri = "ws://127.0.0.1:8001/ws/audio"
    async with websockets.connect(uri, max_size=10_000_000) as ws:
        print("✅ Connected to server")

        audio, sr = sf.read(file_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            raise ValueError(f"Expected {SAMPLE_RATE}Hz audio. Got {sr}Hz. Resample first.")

        send_task = asyncio.create_task(sender(ws, audio))
        recv_task = asyncio.create_task(receiver(ws))
        await asyncio.gather(send_task, recv_task)
        await ws.close()
        print("✅ Finished streaming")

if __name__ == "__main__":
    asyncio.run(run("sample_16k.wav"))