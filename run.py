#!/usr/bin/env python
"""
run.py

Main execution file for text generation using the fine-tuned model.
Usage: python run.py "Your input prompt here"
Output: Returns generated text as string to stdout
"""

import sys
import os
import json
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.text_generator import TextGenerator
from src.utils import setup_environment

def load_config(config_path: str) -> dict:
    """Load generation configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Generate text using fine-tuned model")
    parser.add_argument(
        "prompt",
        type=str,
        help="Input prompt for text generation"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/generation_config.json",
        help="Path to generation configuration file"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Override model path from config"
    )
    parser.add_argument(
        "--max_words",
        type=int,
        help="Override maximum word count from config"
    )
    parser.add_argument(
        "--min_words",
        type=int,
        help="Override minimum word count from config"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Override temperature from config"
    )
    parser.add_argument(
        "--save_output",
        action="store_true",
        help="Save output to outputs/ directory with timestamp"
    )
    
    args = parser.parse_args()
    
    # Setup environment (suppress setup messages for clean output)
    setup_environment(verbose=False)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.model_path:
        config["model_path"] = args.model_path
    if args.max_words:
        config["generation"]["max_words"] = args.max_words
    if args.min_words:
        config["generation"]["min_words"] = args.min_words
    if args.temperature:
        config["generation"]["temperature"] = args.temperature
    
    # Initialize text generator
    try:
        generator = TextGenerator(config)
    except Exception as e:
        print(f"❌ Failed to initialize generator: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Generate text
    try:
        generated_text = generator.generate(args.prompt)
        
        # Output to stdout (clean for piping/capturing)
        print(generated_text)
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path("outputs") / f"generated_{timestamp}.txt"
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Prompt: {args.prompt}\\n\\n")
            f.write(f"Generated Text:\\n{generated_text}")
            
            # Write info to stderr to not pollute stdout        
    except Exception as e:
        print(f"Error generating text: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()