import os
import numpy as np
from faster_whisper import WhisperModel

MODEL_SIZE = os.getenv("WHISPER_MODEL", "small")   
DEVICE = os.getenv("WHISPER_DEVICE", "cpu")        
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
CPU_THREADS = int(os.getenv("WHISPER_CPU_THREADS", max(1, (os.cpu_count() or 4) - 1)))
NUM_WORKERS = int(os.getenv("WHISPER_NUM_WORKERS", "1"))

WORD_TS = os.getenv("WHISPER_WORD_TIMESTAMPS", "1").lower() in {"1", "true", "yes"}
BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "2"))
BEST_OF = int(os.getenv("WHISPER_BEST_OF", str(max(1, BEAM_SIZE))))

model = WhisperModel(
    MODEL_SIZE,
    device=DEVICE,
    compute_type=COMPUTE_TYPE,
    cpu_threads=CPU_THREADS,
    num_workers=NUM_WORKERS,
)

def transcribe_words(audio_np: np.ndarray, sample_rate: int, language: str = "en", initial_prompt: str | None = None):
    """
    Transcribe a window and return per-word tokens with local start/end times.
    [{text, start, end, prob}]
    """
    if audio_np is None or len(audio_np) == 0:
        return []

    if audio_np.dtype != np.float32:
        audio_np = audio_np.astype(np.float32)

    segments, _ = model.transcribe(
        audio_np,
        language=language,
        beam_size=BEAM_SIZE,
        best_of=BEST_OF,
        temperature=0.0,
        word_timestamps=WORD_TS,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=150),
        condition_on_previous_text=True,  
        initial_prompt=initial_prompt,    
    )

    words = []
    for seg in segments:
        if getattr(seg, "words", None):
            for w in seg.words:
                words.append({
                    "text": w.word,  
                    "start": w.start,
                    "end": w.end,
                    "prob": getattr(w, "probability", None),
                })
        else:
            words.append({
                "text": seg.text,
                "start": seg.start,
                "end": seg.end,
                "prob": None,
            })
    return words