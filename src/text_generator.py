"""
text_generator.py

Text generation module using the fine-tuned model.
Implements model caching to avoid reloading on each run.
"""

import torch
import os
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM
from .utils import count_words, ensure_complete_sentence

# Global model cache
_model_cache = {}
_cache_file = "models/.model_cache.pkl"

class TextGenerator:
    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._load_model()
        
    def _load_model(self):
        """Load the fine-tuned model with caching"""
        model_path = self.config["model_path"]
        
        # Check if model is already in memory cache
        if model_path in _model_cache:
            self.model = _model_cache[model_path]['model']
            self.tokenizer = _model_cache[model_path]['tokenizer']
            return
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                        
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            
            # Cache the model in memory
            _model_cache[model_path] = {
                'model': self.model,
                'tokenizer': self.tokenizer
            }
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
        
    def generate(self, prompt: str) -> str:
        """Generate text with fixed word count 2300–2700 words."""
        
        gen_config = self.config["generation"]

        # -----------------------------------------
        # 1. FIXED TARGET WORD COUNT
        # -----------------------------------------
        target_words = (2300, 2700)
        max_tokens = int(target_words[1] * 1.6)  # safe margin

        # Format chat template
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)

        # -----------------------------------------
        # 2. INITIAL GENERATION
        #-----------------------------------------
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.8,
                do_sample=True,
                top_p=0.95,
                top_k=50,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                early_stopping=True,
                length_penalty=1.0
            )

        # Decode response (exclude input prompt)
        full_response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()

        # -----------------------------------------
        # 3. CLEAN REPETITION LINES
        # -----------------------------------------
        lines = full_response.split('\n')
        cleaned_lines = []
        seen_lines = set()

        for line in lines:
            line = line.strip()
            if line and line not in seen_lines:
                cleaned_lines.append(line)
                seen_lines.add(line)
            elif not line:
                cleaned_lines.append(line)

        full_response = '\n'.join(cleaned_lines)

        # -----------------------------------------
        # 4. ENFORCE FIXED WORD COUNT RANGE
        # -----------------------------------------
        word_count = count_words(full_response)

        # Case 1: Already within target range
        if target_words[0] <= word_count <= target_words[1]:
            return ensure_complete_sentence(full_response).strip()

        # Case 2: Too long → trim
        if word_count > target_words[1]:
            words = full_response.split()
            trimmed = ' '.join(words[:target_words[1]])
            return ensure_complete_sentence(trimmed).strip()

        # Case 3: Too short → regenerate up to 2 times
        attempts = 0
        while word_count < target_words[0] and attempts < 2:
            attempts += 1

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.8,
                    do_sample=True,
                    top_p=0.95,
                    top_k=50,
                    repetition_penalty=1.15,
                    no_repeat_ngram_size=3,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            full_response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()

            word_count = count_words(full_response)

        # Case 4: If STILL short → force continuation/padding
        if word_count < target_words[0]:
            continuation_prompt = (
                full_response +
                ""
            )

            continuation_inputs = self.tokenizer(continuation_prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **continuation_inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.8,
                    do_sample=True,
                    top_p=0.95,
                    top_k=50,
                    repetition_penalty=1.15,
                    no_repeat_ngram_size=3,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

            # Final word count check & trim if needed
            words = full_response.split()
            if len(words) > target_words[1]:
                full_response = ' '.join(words[:target_words[1]])

        # Final clean return
        return ensure_complete_sentence(full_response).strip()
