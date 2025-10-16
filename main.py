import os
import time
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio

from transcriber import transcribe_words

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SAMPLE_RATE = 16000

CHUNK_DURATION = float(os.getenv("STREAM_CHUNK_SECONDS", "2.0"))  
OVERLAP = float(os.getenv("STREAM_OVERLAP", "0.5"))               
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
STEP_SIZE = int(CHUNK_SIZE * (1 - OVERLAP))                       
STRIDE_SECONDS = STEP_SIZE / SAMPLE_RATE

SAFE_START = (CHUNK_DURATION - STRIDE_SECONDS) / 2.0  
SAFE_END = SAFE_START + STRIDE_SECONDS                

def select_safe_center(words, global_offset_sec):
    region_start = global_offset_sec + SAFE_START
    region_end = global_offset_sec + SAFE_END
    picked = []
    for w in words:
        if w["start"] is None or w["end"] is None:
            continue
        center_global = (w["start"] + w["end"]) / 2.0 + global_offset_sec
        if region_start <= center_global <= region_end:
            picked.append(w)
    return picked

def words_to_text(words):
    return "".join(w["text"] for w in words).strip()

def clip_context(ctx: str, max_chars: int = 220) -> str:
    if len(ctx) <= max_chars:
        return ctx
    return ctx[-max_chars:]

@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🔗 Client connected")

    buffer = np.array([], dtype=np.float32)
    prev_words = None
    prev_offset_sec = 0.0
    chunk_index = 0

    context_text = ""
    last_sent = ""  

    async def transcribe_window(window_np, prompt: str | None):
        start = time.perf_counter()
        words = await asyncio.to_thread(transcribe_words, window_np, SAMPLE_RATE, initial_prompt=prompt)
        dur = (time.perf_counter() - start)
        rtf = dur / (CHUNK_DURATION if CHUNK_DURATION > 0 else 1e-9)
        print(f"⚙️  decode {CHUNK_DURATION:.2f}s -> {dur*1000:.0f} ms (RTF={rtf:.2f}x)")
        return words

    async def emit(text: str):
        nonlocal context_text, last_sent
        if not text:
            return
        if text == last_sent:
            return
        await websocket.send_text(text)
        last_sent = text
        context_text = clip_context((context_text + " " + text).strip())
        print(f"📝 Committed: {text}")

    async def flush_tail(final_padding=False):
        nonlocal prev_words, prev_offset_sec, buffer, chunk_index

        while len(buffer) >= CHUNK_SIZE:
            window = buffer[:CHUNK_SIZE]
            offset_sec = chunk_index * STRIDE_SECONDS
            words = await transcribe_window(window, prompt=clip_context(context_text))

            if prev_words is not None:
                to_commit = select_safe_center(prev_words, prev_offset_sec)
                await emit(words_to_text(to_commit))

            prev_words = words
            prev_offset_sec = offset_sec
            buffer = buffer[STEP_SIZE:]
            chunk_index += 1

        if prev_words is not None:
            to_commit = select_safe_center(prev_words, prev_offset_sec)
            await emit(words_to_text(to_commit))

        if final_padding and len(buffer) > 0:
            padded = np.zeros(CHUNK_SIZE, dtype=np.float32)
            padded[: len(buffer)] = buffer
            offset_sec = chunk_index * STRIDE_SECONDS
            words = await transcribe_window(padded, prompt=clip_context(context_text))
            await emit(words_to_text(select_safe_center(words, offset_sec)))

    try:
        while True:
            msg = await websocket.receive()

            if "text" in msg and msg["text"] is not None:
                if msg["text"] == "EOS":
                    await flush_tail(final_padding=True)
                    await websocket.send_text("DONE")
                    print("✅ Done + flushed")
                    break
                continue

            data = msg.get("bytes", None)
            if data is None:
                continue

            audio_chunk = np.frombuffer(data, dtype=np.float32)
            buffer = np.concatenate([buffer, audio_chunk])

            while len(buffer) >= CHUNK_SIZE:
                window = buffer[:CHUNK_SIZE]
                offset_sec = chunk_index * STRIDE_SECONDS

                words = await transcribe_window(window, prompt=clip_context(context_text))

                if prev_words is not None:
                    to_commit = select_safe_center(prev_words, prev_offset_sec)
                    await emit(words_to_text(to_commit))

                prev_words = words
                prev_offset_sec = offset_sec

                buffer = buffer[STEP_SIZE:]
                chunk_index += 1

    except WebSocketDisconnect:
        print("🔌 Client disconnected")
    except Exception as e:
        print("❌ Connection error:", repr(e))
        try:
            await flush_tail(final_padding=False)
        except Exception:
            pass

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True)