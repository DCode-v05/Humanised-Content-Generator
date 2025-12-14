"""
trainer.py

Model training module using QLoRA fine-tuning.
"""

import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from .utils import clear_gpu_cache

class ModelTrainer:
    def __init__(self, config: dict):
        self.config = config
        self.tokenizer = None
        self.model = None
        
    def get_tokenizer(self, model_id: str):
        """Load and configure tokenizer"""
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "right"  # Important for causal LM training
        return tok

    def get_model_4bit(self, model_id: str):
        """Load model with 4-bit quantization"""
        quant_config = self.config["model"]["quantization"]
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quant_config["load_in_4bit"],
            bnb_4bit_use_double_quant=quant_config["bnb_4bit_use_double_quant"],
            bnb_4bit_quant_type=quant_config["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=getattr(torch, quant_config["bnb_4bit_compute_dtype"]),
            llm_int8_enable_fp32_cpu_offload=quant_config["llm_int8_enable_fp32_cpu_offload"]
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            dtype=getattr(torch, quant_config["bnb_4bit_compute_dtype"]),
            trust_remote_code=True,
        )
        return model

    def add_lora(self, model):
        """Add LoRA adapters to model"""
        lora_config_dict = self.config["lora"]
        
        lora_config = LoraConfig(
            r=lora_config_dict["r"],
            lora_alpha=lora_config_dict["lora_alpha"],
            lora_dropout=lora_config_dict["lora_dropout"],
            bias=lora_config_dict["bias"],
            task_type=lora_config_dict["task_type"],
            target_modules=lora_config_dict["target_modules"],
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return model

    def formatting_func(self, example):
        """Format example for training"""
        instr = example["instruction"]
        resp = example["response"]
        
        # Data validation
        if not instr or not resp:
            return {"text": ""}
            
        instr = instr.strip()
        resp = resp.strip()

        messages = [
            {"role": "user", "content": instr},
            {"role": "assistant", "content": resp},
        ]
        
        try:
            chat_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            
            # Length validation for training stability
            max_seq_length = self.config["data"]["max_seq_length"]
            if len(self.tokenizer.encode(chat_text)) <= max_seq_length:
                return {"text": chat_text}
            else:
                return {"text": ""}  # Skip if too long
                
        except Exception as e:
            print(f"⚠️ Warning: Skipping malformed example: {e}")
            return {"text": ""}

    def preprocess_dataset(self, dataset):
        """Preprocess dataset for training"""
        formatted = dataset.map(self.formatting_func, remove_columns=dataset.column_names)
        # Filter out empty texts
        filtered = formatted.filter(lambda x: x["text"] and x["text"].strip())
        return filtered

    def train(self):
        """Run the training process"""
        
        # Setup output directory
        output_dir = self.config["training"]["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        
        print("📂 Loading dataset from disk...")
        data_dir = self.config["data"]["data_dir"]
        ds = load_from_disk(data_dir)
        train_ds = ds["train"]
        val_ds = ds["validation"]
        
        print(f"🔤 Loading tokenizer: {self.config['model']['model_id']}")
        self.tokenizer = self.get_tokenizer(self.config["model"]["model_id"])
        
        print("🧠 Loading 4-bit base model...")
        clear_gpu_cache()
        
        self.model = self.get_model_4bit(self.config["model"]["model_id"])
        self.model = self.add_lora(self.model)
        
        # Enable gradient computation for LoRA parameters
        self.model.train()
        
        print("📝 Preprocessing datasets...")
        train_ds_formatted = self.preprocess_dataset(train_ds)
        val_ds_formatted = self.preprocess_dataset(val_ds)
        
        print(f"📊 Training examples: {len(train_ds_formatted)}")
        print(f"📊 Validation examples: {len(val_ds_formatted)}")
        
        # Setup training arguments
        training_config = self.config["training"]
        training_args = TrainingArguments(
            output_dir=training_config["output_dir"],
            per_device_train_batch_size=training_config["per_device_train_batch_size"],
            per_device_eval_batch_size=training_config["per_device_eval_batch_size"],
            gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
            num_train_epochs=training_config["num_train_epochs"],
            learning_rate=training_config["learning_rate"],
            lr_scheduler_type=training_config["lr_scheduler_type"],
            warmup_ratio=training_config["warmup_ratio"],
            logging_steps=training_config["logging_steps"],
            eval_strategy=training_config["eval_strategy"],
            eval_steps=training_config["eval_steps"],
            save_strategy=training_config["save_strategy"],
            save_steps=training_config["save_steps"],
            save_total_limit=training_config["save_total_limit"],
            bf16=training_config["bf16"] and torch.cuda.is_available(),
            gradient_checkpointing=training_config["gradient_checkpointing"],
            max_grad_norm=training_config["max_grad_norm"],
            dataloader_num_workers=training_config["dataloader_num_workers"],
            remove_unused_columns=training_config["remove_unused_columns"],
            report_to=training_config["report_to"],
        )
        
        print("🚀 Starting SFTTrainer...")
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_ds_formatted,
            eval_dataset=val_ds_formatted,
            args=training_args,
        )
        
        trainer.train()
        
        print("💾 Saving LoRA adapter + tokenizer...")
        trainer.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"✅ Saved to {output_dir}")