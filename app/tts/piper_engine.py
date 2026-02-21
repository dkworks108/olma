import queue
import time
import re
import numpy as np
from piper import PiperVoice
import sounddevice as sd


# ---------------------------------------------------------------------------
# Tuning constants — tweak these without touching any logic
# ---------------------------------------------------------------------------

BLOCKSIZE            = 256    # ~5.8ms per callback @44100Hz — very smooth
PRE_SPEECH_MS        = 180    # natural pause before speaking starts
TRAILING_SILENCE_MS  = 280    # prevents last syllable from being clipped
SENTENCE_PAUSE_MS    = 160    # pause injected at . ! ?
COMMA_PAUSE_MS       = 80     # pause injected at ,  ;  :
INTER_CHUNK_MS       = 40     # default gap between synthesis chunks
VOLUME_SCALE         = 0.88   # slight reduction avoids clipping (0.0–1.0)
NOISE_FLOOR          = 1e-4   # silence gate threshold


# ---------------------------------------------------------------------------
# Text pre-processing
# ---------------------------------------------------------------------------

def _preprocess(text: str) -> str:
    """Clean and normalise text so Piper speaks naturally."""
    text = text.strip()

    # Expand common abbreviations
    abbrevs = {
        r"\bAI\b":    "Artificial Intelligence",
        r"\bAPI\b":   "A P I",
        r"\bURL\b":   "U R L",
        r"\bOS\b":    "Operating System",
        r"\bRAM\b":   "ram",
        r"\bCPU\b":   "C P U",
        r"\bGPU\b":   "G P U",
        r"\bLLM\b":   "L L M",
        r"\bTTS\b":   "text to speech",
        r"\bSTT\b":   "speech to text",
        r"\betc\.\b": "et cetera",
        r"\bvs\.\b":  "versus",
        r"\bDr\.\b":  "Doctor",
        r"\bMr\.\b":  "Mister",
        r"\bMrs\.\b": "Misses",
    }
    for pattern, replacement in abbrevs.items():
        text = re.sub(pattern, replacement, text)

    # Convert small integers to words
    _ones = ["", "one", "two", "three", "four", "five", "six", "seven",
             "eight", "nine", "ten", "eleven", "twelve", "thirteen",
             "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
    _tens = ["", "", "twenty", "thirty", "forty", "fifty",
             "sixty", "seventy", "eighty", "ninety"]

    def num_to_words(n: int) -> str:
        if n == 0:
            return "zero"
        if n < 20:
            return _ones[n]
        if n < 100:
            t = _tens[n // 10]
            o = _ones[n % 10]
            return t + ("-" + o if o else "")
        if n < 1000:
            rest = n % 100
            return _ones[n // 100] + " hundred" + (" and " + num_to_words(rest) if rest else "")
        if n < 10000:
            rest = n % 1000
            return _ones[n // 1000] + " thousand" + (" " + num_to_words(rest) if rest else "")
        return str(n)  # too large — leave as-is

    def replace_num(m):
        return num_to_words(int(m.group()))

    text = re.sub(r"\b\d+\b", replace_num, text)

    # Strip markdown
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
    text = re.sub(r"`{1,3}(.*?)`{1,3}", r"\1", text)
    text = re.sub(r"#{1,6}\s*", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)

    # Normalise whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", " ", text)

    return text.strip()


def _split_sentences(text: str) -> list[tuple[str, int]]:
    """
    Split text into (sentence, pause_ms) tuples.
    Sentence-ending punctuation gets SENTENCE_PAUSE_MS,
    commas/semicolons get COMMA_PAUSE_MS, everything else INTER_CHUNK_MS.
    """
    # First split on sentence boundaries
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    result = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Further split on commas within a sentence
        sub_parts = re.split(r"(?<=[,;:])\s+", part)
        for i, sub in enumerate(sub_parts):
            sub = sub.strip()
            if not sub:
                continue
            is_last_sub = (i == len(sub_parts) - 1)
            if sub[-1] in ".!?":
                result.append((sub, SENTENCE_PAUSE_MS))
            elif sub[-1] in ",;:":
                result.append((sub, COMMA_PAUSE_MS))
            elif is_last_sub and part[-1] in ".!?":
                result.append((sub, SENTENCE_PAUSE_MS))
            else:
                result.append((sub, INTER_CHUNK_MS))

    return result if result else [(text, INTER_CHUNK_MS)]


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _to_float32(audio_int16: np.ndarray) -> np.ndarray:
    """int16 PCM → float32 [-1, 1], volume-scaled and clipped."""
    f = audio_int16.astype(np.float32) / 32768.0
    f *= VOLUME_SCALE
    return np.clip(f, -1.0, 1.0)


def _trim_silence(audio: np.ndarray, threshold: float = NOISE_FLOOR) -> np.ndarray:
    """Strip near-silent leading and trailing frames."""
    mask = np.abs(audio) > threshold
    if not np.any(mask):
        return audio
    first = int(np.argmax(mask))
    last  = int(len(mask) - np.argmax(mask[::-1]))
    return audio[first:last]


def _silence(sample_rate: int, duration_ms: int) -> np.ndarray:
    """Generate float32 silence array of given duration."""
    n = max(1, int(sample_rate * duration_ms / 1000))
    return np.zeros(n, dtype=np.float32)


def _fade(audio: np.ndarray, sample_rate: int, fade_ms: int = 8) -> np.ndarray:
    """
    Apply a short linear fade-in and fade-out.
    Eliminates click/pop artefacts at chunk boundaries.
    """
    n = min(int(sample_rate * fade_ms / 1000), len(audio) // 4)
    if n < 2:
        return audio
    ramp  = np.linspace(0.0, 1.0, n, dtype=np.float32)
    audio = audio.copy()
    audio[:n]  *= ramp
    audio[-n:] *= ramp[::-1]
    return audio


# ---------------------------------------------------------------------------
# PiperEngine
# ---------------------------------------------------------------------------

class PiperEngine:
    """
    Human-like, stable, accurate TTS engine wrapping Piper.

    Design
    ------
    1. Full synthesis before stream open — no timing glitches mid-speech.
    2. Sentence + comma pause injection — natural speech rhythm.
    3. Per-chunk silence gate + fade — clean audio, no clicks.
    4. Small blocksize (256) — smooth, low-latency playback.
    5. Volume scaling + clipping — prevents distortion.
    6. Trailing silence — last syllable never clipped.
    """

    def __init__(self, model_path: str):
        self.voice       = PiperVoice.load(model_path)
        self.blocksize   = BLOCKSIZE
        self._q: queue.Queue = queue.Queue(maxsize=1000)
        self.stream      = None
        self._done       = False
        self.sample_rate: int | None = None

    # ------------------------------------------------------------------
    # sounddevice callback
    # ------------------------------------------------------------------

    def _callback(self, outdata, frames, time_info, status):
        outdata.fill(0)
        needed = frames
        pos    = 0

        while needed > 0:
            try:
                chunk = self._q.get_nowait()
            except queue.Empty:
                break                      # underrun → output silence for remaining frames

            if chunk is None:              # sentinel → done
                self._done = True
                break

            take = min(len(chunk), needed)
            outdata[pos:pos + take, 0] = chunk[:take]
            needed -= take
            pos    += take

            if take < len(chunk):          # leftover → push back
                self._q.put(chunk[take:])
                break

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def speak(self, text: str):
        """Synthesise and play text with human-like pacing and accuracy."""
        text = _preprocess(text)
        if not text:
            return

        self._done       = False
        self.sample_rate = None
        segments         = _split_sentences(text)

        # ── Step 1: Synthesise everything into memory ────────────────
        audio_plan: list[np.ndarray] = []

        for sentence, pause_ms in segments:
            sentence_chunks: list[np.ndarray] = []

            for chunk in self.voice.synthesize(sentence):
                if self.sample_rate is None:
                    self.sample_rate = chunk.sample_rate

                audio = _to_float32(chunk.audio_int16_array)
                audio = _trim_silence(audio)
                audio = _fade(audio, self.sample_rate)

                if len(audio) > 0:
                    sentence_chunks.append(audio)

            if not sentence_chunks:
                continue

            # Merge all chunks within one sentence (seamless flow)
            sentence_audio = np.concatenate(sentence_chunks)
            audio_plan.append(sentence_audio)

            # Inject pause after sentence/clause
            if self.sample_rate:
                audio_plan.append(_silence(self.sample_rate, pause_ms))

        if not audio_plan or self.sample_rate is None:
            return

        # ── Step 2: Open stream ONCE with correct sample rate ────────
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.blocksize,
            callback=self._callback,
        )

        
        time.sleep(PRE_SPEECH_MS / 1000)

        
        for audio in audio_plan:
            self._q.put(audio)

     
        self._q.put(_silence(self.sample_rate, TRAILING_SILENCE_MS))

        
        self._q.put(None)

       
        self.stream.start()

        
        while not self._done or not self._q.empty():
            time.sleep(0.015)

        time.sleep(0.06)  # small buffer after sentinel consumed
        self.stop()

    def stop(self):
        """Stop playback immediately and reset state."""
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

        # Drain any remaining queue items
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                break

        self._done = True


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python piper_engine.py <model_path> [optional text]")
        sys.exit(1)

    model = sys.argv[1]
    text  = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else (
        "Hello! I am Sunny, your offline AI assistant. "
        "I can answer questions, help with tasks, and much more. "
        "How can I help you today?"
    )
    

    engine = PiperEngine(model)
    print(f"Speaking: {text}\n")
    engine.speak(text)
    print("Done.")