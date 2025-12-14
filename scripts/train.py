#!/usr/bin/env python
"""
finetune_mixtral_qLoRA.py

QLoRA fine-tuning of Mixtral 8x7B Instruct (or any similar chat model)
on a dataset of (instruction, response) pairs.

Dataset must come from prepare_datasets.py:
  - columns: "instruction", "response"
  - splits: "train", "validation"

Usage example:
    python finetune_mixtral_qLoRA.py \
        --data_dir data/processed_essay_pairs \
        --output_dir models/mixtral-essay-lora
"""

import sys
# Add custom package path for trl if installed in custom location
sys.path.insert(0, r'E:\python_user\lib\site-packages')

# Set HuggingFace cache directory to E: drive (has more space)
import os
os.environ['HF_HOME'] = r'E:\huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = r'E:\huggingface_cache\transformers'
os.environ['HF_DATASETS_CACHE'] = r'E:\huggingface_cache\datasets'

import argparse
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


def get_tokenizer(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Critical for training stability
    tok.padding_side = "right"  # Important for causal LM training
    return tok


def get_model_4bit(model_id: str):
    # Proper 4-bit quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_enable_fp32_cpu_offload=True  # Enable CPU offloading
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,  # Fixed: use dtype instead of torch_dtype
        trust_remote_code=True,
    )
    return model


def add_lora(model):
    """
    Generic LoRA config for decoder-only transformers.

    Mixtral uses standard transformer naming, so targeting q/k/v/o + MLP works.
    Adjust target_modules if you later inspect the model and see different names.
    """
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="mistralai/Mixtral-8x7B-Instruct-v0.1",
        help="Base chat model to fine-tune (default: Mixtral 8x7B Instruct)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed_essay_pairs",
        help="Path to HF dataset saved by prepare_datasets.py",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/mixtral-essay-lora",
        help="Where to save the LoRA adapter",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=4096,
        help="Max tokens per example",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Per-device train batch size",
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=8,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for LoRA",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("📂 Loading dataset from disk...")
    ds = load_from_disk(args.data_dir)
    train_ds = ds["train"]
    val_ds = ds["validation"]

    print("🔤 Loading tokenizer:", args.model_id)
    tokenizer = get_tokenizer(args.model_id)

    print("🧠 Loading 4-bit base model...")
    # Clear GPU cache before loading model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    model = get_model_4bit(args.model_id)
    model = add_lora(model)
    
    # Enable gradient computation for LoRA parameters
    model.train()

    # Turn (instruction, response) → chat-formatted text using model's template
    def formatting_func(example):
        # Handle single example (not batch)
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
            chat_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            
            # Length validation for training stability
            if len(tokenizer.encode(chat_text)) <= args.max_seq_length:
                return {"text": chat_text}
            else:
                return {"text": ""}  # Skip if too long
                
        except Exception as e:
            print(f"⚠️ Warning: Skipping malformed example: {e}")
            return {"text": ""}

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=20,
        eval_strategy="steps",  # Fixed: was evaluation_strategy
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        max_grad_norm=1.0,  # Gradient clipping for stability
        dataloader_num_workers=0,  # Avoid multiprocessing issues
        remove_unused_columns=False,  # Keep all columns for formatting_func
        report_to="none",
    )

    # Preprocess datasets to ensure compatibility
    print("📝 Preprocessing datasets...")
    
    # Apply formatting and filter out empty results
    def preprocess_dataset(dataset):
        formatted = dataset.map(formatting_func, remove_columns=dataset.column_names)
        # Filter out empty texts
        filtered = formatted.filter(lambda x: x["text"] and x["text"].strip())
        return filtered
    
    train_ds_formatted = preprocess_dataset(train_ds)
    val_ds_formatted = preprocess_dataset(val_ds)
    
    print(f"📊 Training examples: {len(train_ds_formatted)}")
    print(f"📊 Validation examples: {len(val_ds_formatted)}")

    print("🚀 Starting SFTTrainer...")
    # Use minimal parameters for your specific TRL version
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds_formatted,
        eval_dataset=val_ds_formatted,
        args=training_args,
    )

    trainer.train()

    print("💾 Saving LoRA adapter + tokenizer...")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"✅ Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
