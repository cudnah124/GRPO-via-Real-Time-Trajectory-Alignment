import torch
from unsloth import FastLanguageModel
from phase1_distillation.prompts import GENERATION_PROMPT

class MathRolloutGenerator:
    def __init__(self, model_id, max_seq_length=4096):
        self.model_id = model_id
        print(f"Loading Model & Tokenizer via Unsloth: {model_id}")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        
        FastLanguageModel.for_inference(self.model)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print("Model loaded successfully for Generation!")

    def generate(self, problem, num_rollouts=4, max_new_tokens=1024):
        messages = [
            {"role": "system", "content": GENERATION_PROMPT},
            {"role": "user", "content": problem}
        ]
        
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
                num_return_sequences=num_rollouts,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True
            )
            
        input_length = inputs['input_ids'].shape[1]
        rollouts = []
        for out in outputs:
            gen_text = self.tokenizer.decode(out[input_length:], skip_special_tokens=True)
            rollouts.append(gen_text)
            
        return rollouts
