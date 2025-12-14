# LLM Fine-tuning Project

A production-ready system for fine-tuning language models using LoRA (Low-Rank Adaptation) with QLoRA quantization. Specialized for generating long-form essays (2300-2700 words) with complete sentence endings.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended: RTX 4090 or equivalent)
- 24GB+ GPU memory for training, 8GB+ for inference

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Danushika06/LLM-Finetuning.git
cd LLM-Finetuning
```

2. **Create virtual environment:**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac  
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Basic Usage

## 🚀 **Main Usage (Primary)**

**Generate text:**
```bash
python run.py "Write a comprehensive essay about artificial intelligence"
```

**Advanced options:**
```bash
# Custom word count
python run.py "Your prompt" --min_words 2000 --max_words 3000

# Custom model and temperature
python run.py "Your prompt" --model_path "models/custom" --temperature 0.9
```

**✨ All generated content is automatically saved to `outputs/` folder with:**
- Timestamp and prompt in filename
- Word count and generation details
- Formatted text with headers

## ⚡ **Fast Generation (Alternative)**

**Server Mode (Fastest - No reload delay):**
```bash
# Terminal 1: Start model server
python scripts/server.py

# Terminal 2: Generate instantly 
python scripts/client.py "Write an essay about AI"
```

**Optimized Single Run:**
```bash
python scripts/fast_run.py "Write an essay about AI"
```

## 📁 Project Structure

```
LLM-Finetuning/
📄 run.py                 # Main execution file ⭐
📄 requirements.txt       # All dependencies
📄 README.md             # Setup and usage instructions
📁 config/               # Configuration files
│   ├── training_config.json
│   ├── merge_config.json
│   └── generation_config.json
📁 scripts/              # Additional scripts
│   ├── train.py             # Model training script
│   ├── merge.py             # LoRA merging script  
│   ├── fast_run.py          # Optimized execution ⚡
│   ├── server.py            # Model server (fastest) 🚀
│   └── client.py            # Fast client for server
📁 src/                  # Source code modules
│   ├── __init__.py
│   ├── utils.py
│   ├── trainer.py
│   ├── model_merger.py
│   └── text_generator.py
📁 models/               # Model weights and adapters
│   ├── mistral-7b-lora/     # LoRA adapters
│   └── mistral-7b-merged/   # Merged standalone model
📁 data/                 # Training data
│   └── processed_essay_pairs/
📁 outputs/              # Generated text outputs
```

## 🎯 Core Features

### Text Generation
- **Word Count Control**: Automatically generates 2300-2700 words
- **Complete Sentences**: Ensures coherent endings, no mid-sentence cuts
- **High Quality**: Fine-tuned on essay datasets for academic writing

### Model Management
- **LoRA Training**: Efficient fine-tuning with 4-bit quantization
- **Model Merging**: Create standalone models without LoRA overhead
- **Flexible Loading**: Automatic fallback between merged and LoRA models

### Production Ready
- **Clean Output**: stdout-only text generation for easy piping
- **Configuration-driven**: JSON configs for all parameters  
- **Error Handling**: Robust error management and logging

## 🛠️ Detailed Usage

### Training a Model

1. **Prepare your data** (if not using existing):
```bash
# Your data should be in format: {"instruction": "...", "response": "..."}
# Place in data/processed_essay_pairs/
```

2. **Configure training** in `config/training_config.json`:
```json
{
  "model": {
    "model_id": "mistralai/Mistral-7B-Instruct-v0.2"
  },
  "training": {
    "num_train_epochs": 2,
    "learning_rate": 1e-4,
    "output_dir": "models/my-fine-tuned-model"
  }
}
```

3. **Run training:**
```bash
python scripts/train.py --config config/training_config.json
```

### Merging Models

**Create a standalone model:**
```bash
python scripts/merge.py --config config/merge_config.json
```

This combines LoRA adapters with the base model for faster inference.

### Configuration

**Generation Config** (`config/generation_config.json`):
```json
{
  "model_path": "models/mistral-7b-merged",
  "generation": {
    "min_words": 2300,
    "max_words": 2700,
    "temperature": 0.7,
    "complete_sentences": true
  }
}
```

**Training Config** (`config/training_config.json`):
```json
{
  "model": {
    "model_id": "mistralai/Mistral-7B-Instruct-v0.2"
  },
  "lora": {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05
  },
  "training": {
    "num_train_epochs": 2,
    "learning_rate": 1e-4,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8
  }
}
```

## 🔧 Advanced Usage

### Integration Examples

**Python script integration:**
```python
import subprocess
import json

def generate_essay(prompt):
    result = subprocess.run(
        ["python", "run.py", prompt], 
        capture_output=True, 
        text=True
    )
    return result.stdout.strip()

essay = generate_essay("Write about climate change")
print(f"Generated {len(essay.split())} words")
```

**Batch processing:**
```bash
# Process multiple prompts
for prompt in "prompt1" "prompt2" "prompt3"; do
    python run.py "$prompt" --save_output
done
```

### Hardware Requirements

**Training:**
- GPU: RTX 4090 (24GB) or equivalent
- RAM: 32GB+ system memory
- Storage: 50GB+ free space

**Inference:**
- GPU: RTX 3080 (10GB) or equivalent  
- RAM: 16GB+ system memory
- Storage: 15GB for model files

### Performance Optimization

**For faster inference:**
1. Use merged models instead of LoRA adapters
2. Adjust `max_words` to reduce generation time
3. Use GPU with higher memory bandwidth

**For training:**
1. Increase `gradient_accumulation_steps` if GPU memory is limited
2. Reduce `max_seq_length` for faster training
3. Use `bf16=true` for better performance

## 📊 Model Specifications

- **Base Model**: Mistral-7B-Instruct-v0.2
- **Fine-tuning**: LoRA with r=16, α=32
- **Quantization**: 4-bit NF4 with double quantization
- **Training Data**: Multi-source essay datasets (~100k examples)
- **Specialization**: Long-form essay generation (2300-2700 words)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -m "Add feature"`
4. Push to branch: `git push origin feature-name`
5. Create a Pull Request

## 🔄 Migration from Old Structure

If you have the old file structure, your models and data are compatible:
- `models/mistral-7b-merged/` → Use with `run.py`  
- `models/mistral-7b-lora/` → Use with `run.py` (auto-detected)
- `data/processed_essay_pairs/` → Use with `train.py`

**Quick migration test:**
```bash
# Test your existing merged model
python run.py "Test prompt" --model_path "models/mistral-7b-merged"

# Test existing LoRA model  
python run.py "Test prompt" --model_path "models/mistral-7b-lora"
``` 
