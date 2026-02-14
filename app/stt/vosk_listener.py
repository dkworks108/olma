import json
import queue
import sounddevice as sd
from vosk import Model, KaldiRecognizer

class VoskListener:
    def __init__(self, model_path, samplerate=16000):
        self.model = Model(model_path)
        self.recognizer = KaldiRecognizer(self.model, samplerate)
        self.q = queue.Queue()
        self.samplerate = samplerate

    def _audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        self.q.put(bytes(indata))

    def listen(self):
        print("🎤 Listening... बोलिए")

        with sd.RawInputStream(
            samplerate=self.samplerate,
            blocksize=8000,
            dtype="int16",
            channels=1,
            callback=self._audio_callback,
        ):
            while True:
                data = self.q.get()
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    text = result.get("text", "").strip()
                    if text:
                        return text
