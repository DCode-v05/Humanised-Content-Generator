# Humanised Content Generator

**A QLoRA fine-tuning and inference setup for Mistral-7B that writes long-form essays (2300–2700 words) which read like a person wrote them and end on a complete sentence.**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white) ![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=flat&logo=huggingface&logoColor=black) ![PEFT](https://img.shields.io/badge/PEFT%20%2F%20LoRA-555555?style=flat) ![bitsandbytes](https://img.shields.io/badge/bitsandbytes%204--bit-555555?style=flat)

## Overview

Standard instruction-tuned LLMs are good at short answers but tend to fall apart on long essays — they drift, repeat themselves, or stop mid-sentence well before the length you asked for. This project fine-tunes **Mistral-7B-Instruct-v0.2** with QLoRA on (instruction, response) essay pairs, then wraps it in a generation pipeline that enforces a target length of 2300–2700 words and guarantees the output stops on a finished sentence rather than a dangling clause.

It's a complete personal project: a training path (QLoRA on a 4-bit base model), a merge step that folds the LoRA adapter back into the base weights for standalone inference, and two ways to run generation — a one-shot CLI and a warm socket server that keeps the model in memory for repeated requests. Everything reads its settings from JSON config files and prints clean text to stdout so it can be piped into other tools.

## Key Features

- **QLoRA fine-tuning** of Mistral-7B-Instruct-v0.2 on a 4-bit NF4 quantized base, so the whole training run fits on a single 24 GB GPU.
- **Length-controlled generation** — every essay is pushed into the 2300–2700 word band with a generate / trim / regenerate / continue loop, not just a `max_new_tokens` cap.
- **Complete-sentence guarantee** — output is walked backward to the last real sentence ending (`.`, `!`, `?`) so it never cuts off mid-thought.
- **Repetition cleanup** — duplicate lines are stripped after generation, on top of `repetition_penalty` and `no_repeat_ngram_size` during sampling.
- **Adapter merging** — `merge_and_unload()` combines the trained LoRA adapter with the base model into a single deployable checkpoint.
- **Warm-server inference** — a threaded socket server holds the model in memory; a thin client sends a prompt and gets text back, avoiding a multi-second reload per call.
- **In-process model cache** — even outside server mode, the loaded model and tokenizer are cached so repeated generations in one process don't reload weights.
- **Config-driven** — generation, training, and merge parameters all live in JSON files, with CLI flags to override model path, temperature, and word limits per run.
- **Clean stdout** — generated text goes to stdout, status and errors to stderr, so the output is safe to capture or pipe.

## How It Works

### Training (QLoRA)

The base model is loaded in 4-bit with `BitsAndBytesConfig` — NF4 quantization, double quantization, `bfloat16` compute dtype, and CPU offload enabled for the fp32 parts. A LoRA adapter is attached with PEFT using rank `r=16`, `alpha=32`, dropout `0.05`, no bias, targeting the attention and MLP projections (`q/k/v/o_proj` and `gate/up/down_proj`).

Training data is a Hugging Face dataset on disk with `instruction` / `response` columns and `train` / `validation` splits. Each pair is rendered through the model's chat template, examples that exceed the max sequence length (4096 tokens) or are empty get filtered out, and the rest are handed to TRL's `SFTTrainer`. Defaults are 2 epochs, learning rate `1e-4` with a cosine schedule and 3% warmup, per-device batch size 1 with gradient accumulation of 8, gradient checkpointing on, and gradient clipping at `1.0`. The output is a saved LoRA adapter plus tokenizer.

Two training entry points exist. `src/trainer.py` (`ModelTrainer`) is the config-driven version used by the project and reads everything from a training config. `scripts/train.py` is a standalone CLI variant whose default `--model_id` is Mixtral-8x7B-Instruct-v0.1, but it's overridable — for the 24 GB single-GPU target this project documents, Mistral-7B is the model that actually fits.

### Merging

`src/model_merger.py` (`ModelMerger`) loads the base model in `float16`, attaches the trained adapter with `PeftModel.from_pretrained`, calls `merge_and_unload()` to bake the LoRA weights into the base, and saves a standalone checkpoint plus tokenizer. After this you can run inference without PEFT in the loop.

### Generation and length control

This is the core of the inference side, in `src/text_generator.py`. The flow per prompt:

1. The prompt is wrapped in the chat template with a generation prompt, then sampled with `max_new_tokens` sized to roughly 1.6× the target word ceiling. Sampling uses `temperature=0.8`, `top_p=0.95`, `top_k=50`, `repetition_penalty=1.15`, and `no_repeat_ngram_size=3`.
2. The raw output is de-duplicated line by line to drop repeated paragraphs.
3. Word count is checked against the 2300–2700 target:
   - **In range** → return after the complete-sentence pass.
   - **Too long** → trim to the word ceiling, then complete-sentence pass.
   - **Too short** → regenerate up to 2 more times; if still short, run a continuation pass that feeds the existing text back in to extend it, then trim if it overshoots.
4. `ensure_complete_sentence()` scans backward to the last sentence-ending punctuation followed by whitespace (or end of text) and cuts there, so the essay always ends cleanly.

### Serving

`scripts/server.py` loads the model once and listens on `localhost:8765`. Each incoming connection is handled on its own thread: it reads the prompt, runs the same `TextGenerator.generate()` pipeline, and writes the text back over the socket. `scripts/client.py` opens a connection, sends the prompt, and streams the response until the socket closes — so generation cost is just the forward pass, with no model reload.

## Highlights

No quality benchmark is shipped with the repo, so there are no accuracy figures — but the design specs are concrete:

- **Target length:** 2300–2700 words, enforced (not just requested), with up to 2 regenerations plus a continuation fallback when output comes in short.
- **Adapter:** LoRA `r=16`, `alpha=32`, dropout `0.05` on 4-bit NF4 + double-quantized weights.
- **Training config:** 4096-token sequences, 2 epochs, lr `1e-4` (cosine, 3% warmup), batch 1 × grad-accum 8, gradient checkpointing, grad clip `1.0`.
- **Sampling:** temp `0.8`, top-p `0.95`, top-k `50`, rep-penalty `1.15`, no-repeat 3-grams.
- **Hardware targets:** inference fits an RTX 3080 (10 GB); training is sized for an RTX 4090 (24 GB).

## Tech Stack

- **Language:** Python 3.8+
- **ML framework:** PyTorch
- **Models / fine-tuning:** Hugging Face Transformers, PEFT (LoRA), TRL (`SFTTrainer`), bitsandbytes (4-bit NF4 QLoRA), Accelerate, Safetensors, Datasets
- **Base model:** `mistralai/Mistral-7B-Instruct-v0.2`
- **Infra:** Git LFS for model/data artifacts, JSON config files, a plain TCP socket server/client for warm inference

## Getting Started

### Prerequisites

- Python 3.8+ and a CUDA-capable GPU (10 GB+ for inference, 24 GB for training)
- The base model accessible from Hugging Face (and an HF login if it's gated)
- Your own `config/*.json` files — these are git-ignored, so they aren't in the repo and you'll need to supply them (`generation_config.json`, `merge_config.json`, `training_config.json`)
- A prepared dataset on disk with `instruction` / `response` columns and `train` / `validation` splits, if you intend to train

### Installation

```bash
git clone https://github.com/DCode-v05/Humanised-Content-Generator.git
cd Humanised-Content-Generator

python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

### Running

One-off generation:

```bash
python run.py "Write a comprehensive essay about artificial intelligence"
```

Warm server mode (two terminals):

```bash
# Terminal 1 — load the model once
python scripts/server.py

# Terminal 2 — send prompts
python scripts/client.py "Write an essay about AI"
```

## Usage

- **Generate (one-shot):** `python run.py "<prompt>"`. Flags let you override config — `--config`, `--model_path`, `--max_words`, `--min_words`, `--temperature`, `--save_output`. Output goes to stdout and is also written to `outputs/generated_<timestamp>.txt`.
- **Generate (optimized single run):** `python scripts/fast_run.py "<prompt>"`, same generator with model caching and `--save_output` to dump to `outputs/`.
- **Generate (server):** start `scripts/server.py`, then call `scripts/client.py "<prompt>" [--port 8765]` as often as you like without reloading.
- **Train:** point `src/trainer.py` / `scripts/train.py` at your dataset and config to produce a LoRA adapter. The standalone CLI takes `--model_id`, `--data_dir`, `--output_dir`, `--epochs`, `--learning_rate`, `--batch_size`, `--grad_accum`, and `--max_seq_length`.
- **Merge:** `python scripts/merge.py --config config/merge_config.json` (or override `--base_model` / `--lora_path` / `--output_path`) to fold the adapter into the base model for standalone inference.

## Project Structure

```
Humanised-Content-Generator/
├── run.py                  # Main CLI: generate one essay, print to stdout, save to outputs/
├── extract_lfs.py          # Helper to expand Git LFS pointer files back to real content
├── requirements.txt        # Torch, Transformers, PEFT, TRL, bitsandbytes, Accelerate, etc.
├── .gitattributes          # Git LFS rules for weights / data / archive files
├── .gitignore              # Ignores config/*.json, data/, model binaries, outputs/
├── scripts/
│   ├── train.py            # Standalone QLoRA training CLI (Mixtral default, overridable)
│   ├── merge.py            # CLI to merge a LoRA adapter into the base model
│   ├── fast_run.py         # Generation with model caching + optional file save
│   ├── server.py           # Threaded socket server, keeps model warm on :8765
│   └── client.py           # Thin client that sends a prompt to the server
└── src/
    ├── __init__.py
    ├── text_generator.py   # Generation + 2300–2700 word enforcement + sentence completion
    ├── trainer.py          # Config-driven QLoRA trainer (ModelTrainer)
    ├── model_merger.py     # PEFT merge_and_unload into a standalone checkpoint
    └── utils.py            # Word counting, sentence completion, GPU cache, env setup
```

Note: `config/`, `models/`, `data/`, and `outputs/` are referenced by the code but git-ignored, so they aren't part of the committed tree — you create them locally.

---

## Contact

<table>
  <tr><td><b>Portfolio:</b> <a href="https://www.denistan.me">Denistan</a></td><td><b>LinkedIn:</b> <a href="https://www.linkedin.com/in/denistanb">denistanb</a></td></tr>
  <tr><td><b>GitHub:</b> <a href="https://github.com/DCode-v05">DCode-v05</a></td><td><b>LeetCode:</b> <a href="https://leetcode.com/u/Denistan_B">Denistan_B</a></td></tr>
  <tr><td colspan="2" align="center"><b>Email:</b> <a href="mailto:denistanb05@gmail.com">denistanb05@gmail.com</a></td></tr>
</table>

Made with ❤️ by **Denistan B**
