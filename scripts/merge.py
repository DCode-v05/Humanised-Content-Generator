#!/usr/bin/env python
"""
merge.py

Model merging script to combine LoRA adapters with base model.
Creates a standalone model for deployment.
"""

import sys
import os
import json
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

from src.model_merger import ModelMerger
from src.utils import setup_environment

def load_config(config_path: str) -> dict:
    """Load merge configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapters with base model")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/merge_config.json",
        help="Path to merge configuration file"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        help="Override base model ID from config"
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        help="Override LoRA adapter path from config"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Override output path from config"
    )
    
    args = parser.parse_args()
    
    setup_environment()
    
    config = load_config(args.config)
    
    if args.base_model:
        config["base_model_id"] = args.base_model
    if args.lora_path:
        config["lora_path"] = args.lora_path
    if args.output_path:
        config["output_path"] = args.output_path
    
    print("🔗 Starting model merging...")
    print(f"📋 Using config: {args.config}")
    print(f"🤖 Base model: {config['base_model_id']}")
    print(f"🎯 LoRA adapters: {config['lora_path']}")
    print(f"💾 Output: {config['output_path']}")
    
    merger = ModelMerger(config)
    merger.merge()
    
    print("✅ Model merging completed successfully!")
    print(f"📦 Merged model saved to: {config['output_path']}")

if __name__ == "__main__":
    main()