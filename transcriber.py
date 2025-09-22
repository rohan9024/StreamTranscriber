from faster_whisper import WhisperModel

model = WhisperModel("small", device="cpu", compute_type="int8")

def transcribe_chunk(audio_np):
    segments, _ = model.transcribe(audio_np, beam_size=1)
    if segments:
        return " ".join([seg.text for seg in segments])
    return ""
