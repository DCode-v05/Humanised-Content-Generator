# Humanised-Content-Generator

QLoRA fine-tuning and inference scripts for Mistral-7B-Instruct-v0.2 that write long essays (2300 to 2700 words) and stop on a finished sentence.

Instruction-tuned models handle short answers fine but drift, repeat themselves or stop mid-sentence on long essays. I fine-tuned the model on (instruction, response) essay pairs, then wrapped generation in a loop that checks the word count and cuts the output at the last full sentence. The repo has the training path, a step that merges the LoRA adapter into the base model, and two ways to generate: a one-shot CLI and a socket server that keeps the model loaded. It ships no model weights, datasets or configs.

## How it works

**Training.** The base model is loaded in 4-bit with `BitsAndBytesConfig` (NF4, double quantization, `bfloat16` compute, fp32 CPU offload). A LoRA adapter is attached with PEFT: `r=16`, `alpha=32`, dropout `0.05`, no bias, on `q/k/v/o_proj` and `gate/up/down_proj`. The data is a Hugging Face dataset on disk with `instruction` and `response` columns and `train` and `validation` splits. Each pair goes through the chat template, examples over the max sequence length (4096 tokens) or empty ones are dropped, and TRL's `SFTTrainer` does the rest. Defaults: 2 epochs, learning rate `1e-4` with a cosine schedule and 3% warmup, batch size 1 with gradient accumulation 8, gradient checkpointing, gradient clipping 1.0.

`scripts/train.py` is the training entry point. Its default `--model_id` is `mistralai/Mixtral-8x7B-Instruct-v0.1`, so pass the Mistral-7B id explicitly. `src/trainer.py` has a config-driven `ModelTrainer` class, but no script in the repo calls it.

**Merging.** `src/model_merger.py` loads the base model in `float16`, attaches the adapter with `PeftModel.from_pretrained`, calls `merge_and_unload()` and saves a standalone checkpoint and tokenizer.

**Generation** (`src/text_generator.py`). For each prompt:

1. Apply the chat template and sample with `max_new_tokens` set to about 1.6 times the word ceiling, using `temperature=0.8`, `top_p=0.95`, `top_k=50`, `repetition_penalty=1.15` and `no_repeat_ngram_size=3`.
2. Remove repeated lines.
3. Check the word count. In range: return. Too long: trim to the ceiling. Too short: regenerate up to 2 more times, then run a continuation pass that feeds the text back in, and trim if it overshoots.
4. `ensure_complete_sentence()` cuts at the last sentence-ending punctuation, so the essay ends cleanly.

**Serving.** `scripts/server.py` loads the model once and listens on `localhost:8765`, one thread per connection. `scripts/client.py` sends a prompt and prints the reply.

Generated text goes to stdout and status and errors go to stderr, so the output can be piped.

No quality benchmark is included, so there are no accuracy figures.

## Stack

Python, PyTorch, Transformers, PEFT, TRL, bitsandbytes, Accelerate, Datasets, Safetensors. Git LFS rules for weights and data are in `.gitattributes`.

## Setup

You need Python and a CUDA GPU, with access to the base model on Hugging Face (and a login if it is gated).

```bash
git clone https://github.com/DCode-v05/Humanised-Content-Generator.git
cd Humanised-Content-Generator
python -m venv venv
venv\Scripts\activate        # Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

`config/*.json` is git-ignored, so create your own. From the code, `config/generation_config.json` needs a `model_path` and a `generation` block (`max_words`, `min_words`, `temperature` and the other sampling settings), and `config/merge_config.json` needs `base_model_id`, `lora_path` and `output_path`.

Generate one essay:

```bash
python run.py "Write a comprehensive essay about artificial intelligence"
```

`run.py` accepts `--config`, `--model_path`, `--max_words`, `--min_words` and `--temperature`. It prints the essay and always writes a copy to `outputs/generated_<timestamp>.txt`. `scripts/fast_run.py` does the same with model caching and only saves with `--save_output`.

Server mode, in two terminals:

```bash
python scripts/server.py
python scripts/client.py "Write an essay about AI"
```

Merge an adapter into the base model:

```bash
python scripts/merge.py --config config/merge_config.json
```

Train, with your dataset in place (the `--data_dir` default is `data/processed_essay_pairs`):

```bash
python scripts/train.py --model_id mistralai/Mistral-7B-Instruct-v0.2 --output_dir models/mistral-essay-lora
```

Other flags: `--epochs`, `--learning_rate`, `--batch_size`, `--grad_accum`, `--max_seq_length`.

## Structure

```
run.py                  generate one essay
scripts/
  train.py              QLoRA training CLI
  merge.py              merge adapter into base model
  fast_run.py           generation with model caching
  server.py, client.py  warm-model socket server and client
src/
  text_generator.py     generation, word-count control, sentence completion
  trainer.py            ModelTrainer (not called by any script)
  model_merger.py       merge_and_unload wrapper
  utils.py              word count, sentence completion, GPU cache, env setup
```

`config/`, `models/`, `data/` and `outputs/` are used by the code but git-ignored, so create them locally.

## Contact

[Portfolio](https://www.denistan.me) · [LinkedIn](https://www.linkedin.com/in/denistanb) · [GitHub](https://github.com/DCode-v05) · denistanb05@gmail.com
