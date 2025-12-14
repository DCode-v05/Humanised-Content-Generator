#!/usr/bin/env python
"""
fast_run.py

Optimized version of run.py with better generation parameters and model caching.
Usage: python fast_run.py "Your input prompt here"
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
    parser = argparse.ArgumentParser(description="Fast text generation")
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
    
    # Initialize text generator (will use caching)
    generator = TextGenerator(config)
    
    # Generate text
    try:
        generated_text = generator.generate(args.prompt)
        
        # Output to stdout (clean for piping/capturing)
        print(generated_text)
        
        # Optionally save to file
        if args.save_output:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = Path("outputs") / f"generated_{timestamp}.txt"
            output_file.parent.mkdir(exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"Prompt: {args.prompt}\\n\\n")
                f.write(f"Generated Text:\\n{generated_text}")
            
            # Write info to stderr to not pollute stdout
            print(f"Output saved to: {output_file}", file=sys.stderr)
        
    except Exception as e:
        print(f"Error generating text: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()