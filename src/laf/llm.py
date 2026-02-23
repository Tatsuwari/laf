from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, GenerationConfig
from .config import SystemConfig, GenConfig

class LLM:
    def __init__(self, cfg: SystemConfig):
        self.cfg = cfg

        profile = cfg.model_profiles.get(cfg.active_profile)
        self.model_name = profile.model_name
        self.dtype = profile.dtype
        self.device_map = profile.device_map

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            device_map=self.device_map
        )
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def chat(self, system: str, user: str, gen: GenConfig) -> str:
        messages = [
            {"role": "system", "content": system.strip()},
            {"role": "user", "content": user.strip()},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        generation_config = GenerationConfig(
            max_new_tokens=gen.max_new_tokens,
            do_sample=gen.do_sample,
            temperature=gen.temperature,
            top_p=gen.top_p,
            top_k=gen.top_k if gen.top_k > 0 else None,
            repetition_penalty=gen.repetition_penalty,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        out = self.pipe(prompt, generation_config=generation_config)[0]["generated_text"]
        # Split off assistant if present
        if "<|assistant|>" in out:
            return out.split("<|assistant|>")[-1].strip()
        return out.strip()
