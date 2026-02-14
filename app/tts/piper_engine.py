from piper import PiperVoice
import sounddevice as sd


class PiperEngine:
    """
    Offline Hindi TTS using Piper (Python API)

    Model Name: olima
    Created By: Doresh
    Studio: dkworks108
    """

    def __init__(self, model_path: str):
        """
        model_path example:
        models/piper/hi_IN-priyamvada-medium.onnx
        """
        self.voice = PiperVoice.load(model_path)

    def speak(self, text: str):
        """
        Speak the given text in Hindi.
        Audio is streamed chunk-by-chunk for low latency.
        """
        if not text or not text.strip():
            return

        for chunk in self.voice.synthesize(text):
            sd.play(chunk.audio_int16_array, chunk.sample_rate)
            sd.wait()
