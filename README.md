# Humanised Content Generator

## Project Description
A production-ready system for fine-tuning language models using LoRA (Low-Rank Adaptation) with QLoRA quantization. This project is specialized for generating long-form essays (2300-2700 words) that feel human-written, maintaining coherence and ensuring complete sentence endings.

---

## Project Details

### Problem Statement
Generating high-quality, long-form academic content that maintains coherence over thousands of words is a challenge for standard LLMs. This project addresses the need for a system that can consistently produce essays in the 2300-2700 word range without "hallucinating" abruptly or cutting off mid-sentence, suitable for automated content generation tasks.

### Model Architecture & Fine-tuning
- **Base Model:** Mistral-7B-Instruct-v0.2
- **Method:** QLoRA (4-bit quantization)
- **Adapter Configuration:** 
  - Rank (r): 16
  - Alpha: 32
  - Dropout: 0.05
- **Quantization:** 4-bit NF4 with double quantization

### Core Features
- **Word Count Control:** Automatically generates content within the specific 2300-2700 word range.
- **Complete Sentences:** Implements logic to ensure the generation ends on a complete sentence, avoiding mid-thought cutoffs.
- **Model Merging:** Includes tools to merge LoRA adapters with the base model for standalone, high-performance inference.
- **Production Ready:** Designed with stdout-only text generation for easy piping into other applications, and fully configuration-driven via JSON.

### Performance Optimization
- **Inference:** optimized for RTX 3080 (10GB) or equivalent.
- **Training:** optimized for RTX 4090 (24GB).
- **Fast Mode:** Includes a server-client architecture (`server.py` and `client.py`) to keep the model loaded in memory for instant responses.

---

## Tech Stack
- Python 3.8+
- PyTorch
- Hugging Face Transformers
- PEFT (Parameter-Efficient Fine-Tuning)
- BitsAndBytes (for quantization)
- Safetensors

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/DCode-v05/Humanised-Content-Generator.git
cd Humanised-Content-Generator
```

### 2. Install dependencies
It is recommended to use a virtual environment.
```bash
# Create venv
python -m venv venv

# Activate venv (Windows)
venv\Scripts\activate

# Activate venv (Linux/Mac)
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. Run the Application
**Basic Generation:**
```bash
python run.py "Write a comprehensive essay about artificial intelligence"
```

**Fast Server Mode:**
Terminal 1:
```bash
python scripts/server.py
```
Terminal 2:
```bash
python scripts/client.py "Write an essay about AI"
```

---

## Usage
- **Text Generation:** Use `run.py` for one-off generation or `scripts/fast_run.py` for optimized single runs.
- **Training:** Use `scripts/train.py` with `config/training_config.json` to fine-tune your own models on new data.
- **Merging:** Use `scripts/merge.py` to combine your trained LoRA adapters with the base model.
- **Configuration:** Modify JSON files in the `config/` directory to adjust generation parameters (temperature, max words) or training hyperparameters.

---

## Project Structure
```
Humanised-Content-Generator/
│
├── run.py                 # Main execution script for generation
├── requirements.txt       # Project dependencies
├── extract_lfs.py         # Utility to extract Git LFS objects
├── README.md              # Project documentation
├── config/                # Configuration files (training, merging, generation)
├── scripts/               # Utility scripts
│   ├── train.py           # Model training script
│   ├── merge.py           # Adapter merging script
│   ├── fast_run.py        # Optimized execution script
│   ├── server.py          # Model server for fast inference
│   └── client.py          # Client for server mode
├── src/                   # Source code
│   ├── __init__.py
│   ├── utils.py           # Utility functions
│   ├── trainer.py         # Training logic
│   ├── model_merger.py    # Merging logic
│   └── text_generator.py  # Generation logic
├── models/                # Directory for model weights and adapters
├── data/                  # Directory for training datasets
└── outputs/               # Directory for generated content
```

---

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository
2. Create a new branch:
   ```bash
   git checkout -b feature/your-feature
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add your feature"
   ```
4. Push to your branch:
   ```bash
   git push origin feature/your-feature
   ```
5. Open a pull request describing your changes.

---

## Contact
- **GitHub:** [DCode-v05](https://github.com/DCode-v05)
- **Email:** denistanb05@gmail.com
