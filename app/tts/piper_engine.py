from piper import PiperVoice
import sounddevice as sd
import queue
import time


class PiperEngine:
    def __init__(self, model_path: str, blocksize: int = 2048):
        self.voice = PiperVoice.load(model_path)
        self.audio_queue = queue.Queue(maxsize=200)
        self.blocksize = blocksize
        self.stream = None
        self.finished = False
        self.sample_rate = None

    def _audio_callback(self, outdata, frames, time_info, status):
        outdata.fill(0)

        frames_needed = frames
        pos = 0

        while frames_needed > 0:
            try:
                chunk = self.audio_queue.get_nowait()

                if chunk is None:
                    self.finished = True
                    break

                take = min(len(chunk), frames_needed)
                outdata[pos:pos + take, 0] = chunk[:take] / 32768.0

                frames_needed -= take
                pos += take

                if take < len(chunk):
                    self.audio_queue.put(chunk[take:])
                    break

            except queue.Empty:
                break

    def speak(self, text: str):
        if not text.strip():
            return

        self.finished = False
        self.sample_rate = None

        time.sleep(0.12)  # small natural pause

        for chunk in self.voice.synthesize(text):
            if self.sample_rate is None:
                self.sample_rate = chunk.sample_rate
                self.stream = sd.OutputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype="float32",
                    blocksize=self.blocksize,
                    callback=self._audio_callback,
                )
                self.stream.start()

            self.audio_queue.put(chunk.audio_int16_array)

        self.audio_queue.put(None)

        while not self.finished or not self.audio_queue.empty():
            time.sleep(0.02)

        time.sleep(0.1)
        self.stop()

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
