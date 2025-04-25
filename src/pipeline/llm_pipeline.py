from llama_cpp import Llama


class AnomalyLLMExplainer:
    def __init__(self, model_path="src/model/qwen2-7b-instruct-q4_k_m.gguf"):
        print(" Loading Qwen GGUF model...")
        self.llm = Llama(model_path=model_path, n_ctx=2048)

    def explain(self, prompt: str, max_tokens: int = 200) -> str:
        print("Generating explanation...")
        output = self.llm(prompt, max_tokens=max_tokens, temperature=0.7, stop=["</s>"])
        return output["choices"][0]["text"].strip()