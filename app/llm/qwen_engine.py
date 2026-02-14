import subprocess


class QwenEngine:
    """
    Offline LLM Engine using Qwen 2.5 (GGUF) via llama.cpp

    Model Name: olima
    Created By: Doresh
    Studio: dkworks108
    """

    def __init__(self, model_path: str, llama_bin: str):
        self.model_path = model_path
        self.llama_bin = llama_bin

    def ask(self, user_text: str) -> str:
        if not user_text or not user_text.strip():
            return ""

        # Strong but short system prompt (lightweight)
        prompt = (
            "तुम एक शांत, आत्मविश्वासी, मजबूत हिंदी बोलने वाले AI हो। "
            "जवाब छोटे, साफ और सम्मानजनक हों। "
            "अनावश्यक शब्द मत जोड़ो।\n\n"
            f"प्रश्न: {user_text}\n"
            "उत्तर:"
        )

        cmd = [
            self.llama_bin,
            "-m", self.model_path,

            # 🔑 LIGHTWEIGHT + FAST SETTINGS
            "--ctx-size", "512",
            "--n-predict", "64",

            "--temp", "0.2",
            "--top-p", "0.9",
            "--repeat-penalty", "1.1",

            # 🔑 NON-INTERACTIVE (NO HANG)
            "--simple-io",
            "--no-display-prompt",
            "--log-disable",

            "-p", prompt,
        ]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=180  # realistic CPU timeout
            )
        except subprocess.TimeoutExpired:
            return "माफ़ कीजिए, अभी उत्तर देने में समस्या आ रही है।"

        if result.returncode != 0:
            return "कुछ तकनीकी समस्या आ गई है।"

        output = result.stdout.strip()

        # Final safety cleanup
        if "उत्तर:" in output:
            output = output.split("उत्तर:")[-1].strip()

        return output
