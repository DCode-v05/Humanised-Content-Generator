"""
model_merger.py

Model merging module to combine LoRA adapters with base model.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

class ModelMerger:
    def __init__(self, config: dict):
        self.config = config
        
    def merge(self):
        """Merge LoRA adapters with base model"""
        
        base_model_id = self.config["base_model_id"]
        lora_path = self.config["lora_path"]
        output_path = self.config["output_path"]
        
        print(f"🔤 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        
        print(f"🧠 Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            dtype=torch.float16,
            device_map="auto",
        )
        
        print(f"🔗 Loading LoRA adapters...")
        model = PeftModel.from_pretrained(base_model, lora_path)
        
        print(f"⚡ Merging LoRA with base model...")
        merged_model = model.merge_and_unload()
        
        print(f"💾 Saving merged model to {output_path}...")
        merged_model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)