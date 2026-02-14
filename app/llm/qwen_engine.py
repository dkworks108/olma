import subprocess


class QwenEngine:
    """
    Model Name : olima
    Created By : Doresh
    Company    : dkworks108

    Description:
    Offline Hindi-first reasoning engine built on Qwen (GGUF)
    using llama.cpp backend. Optimized for stability, speed,
    and short confident responses.
    """

    def __init__(self, model_path, llama_bin, ctx=1024):
        self.model_path = model_path
        self.llama_bin = llama_bin
        self.ctx = ctx

        self.system_prompt = (
            "तुम एक शांत, आत्मविश्वासी, मजबूत हिंदी बोलने वाले AI हो। "
            "जवाब छोटे, साफ और सम्मानजनक हों। "
            "तर्क करके उत्तर दो।"
        )

    def ask(self, user_text: str) -> str:
        prompt = (
            "निर्देश:\n"
            "- केवल सरल, शुद्ध हिंदी में उत्तर दो\n"
            "- उत्तर छोटा और स्पष्ट हो\n"
            "- अंग्रेज़ी शब्दों का प्रयोग मत करो\n"
            "- अनावश्यक जानकारी मत जोड़ो\n\n"
            f"प्रश्न: {user_text}\n"
            "उत्तर:"
        )

        cmd = [
            self.llama_bin,
            "-m", self.model_path,
            "-p", prompt,
            "--ctx-size", str(self.ctx),
            "--temp", "0.2",
            "--top-p", "0.85",
            "--repeat-penalty", "1.1",
            "--n-predict", "80",
            "--threads", "6"
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        raw_output = result.stdout.strip()

        if "उत्तर:" in raw_output:
            raw_output = raw_output.split("उत्तर:")[-1].strip()

        return self._clean_hindi_output(raw_output)

    def _clean_hindi_output(self, text: str) -> str:
        if not text:
            return "मैं आपकी सहायता के लिए यहाँ हूँ।"

        blacklist = [
            "machine learning",
            "large language model",
            "ai model",
            "i am",
            "i'm",
            "created by",
            "alibaba",
            "language model"
        ]

        lower = text.lower()
        for bad in blacklist:
            if bad in lower:
                return "मैं एक ऑफलाइन सहायक हूँ और आपकी मदद कर सकता हूँ।"

        words = text.split()
        if len(words) > 35:
            text = " ".join(words[:35])

        if len(text.strip()) < 5:
            return "मैं आपकी सहायता के लिए तैयार हूँ।"

        return text.strip()
