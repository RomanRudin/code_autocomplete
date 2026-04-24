# Python Code Autocomplete — Dual Transformer System

A from-scratch Transformer-based autocomplete system for Python source code,
consisting of **two specialised models** trained on 10 000+ GitHub files.

```
raw_data/          ← your raw .py files from GitHub
  ├── repo1/
  ├── repo2/
  └── ...

data/              ← prepared splits (created by prepare_data.py)
  ├── train/
  ├── val/
  └── test/

checkpoints/       ← best model weights (auto-saved during training)
plots/             ← training metric PNGs (saved every ~20% of epochs)
eval_results/      ← offline evaluation JSON + dashboard PNG
```

---

## Architecture

### TokenModel — next-token / next-word prediction
* **Decoder-only Transformer** (causal self-attention)
* Input : sliding context window of token IDs
* Output: distribution over vocab at each position
* Stopping rule: generates until the first punctuation / whitespace boundary
  → gives you the **next word or sub-expression**

### LineModel — full line completion
* **Encoder-Decoder Transformer** (seq2seq)
* Encoder reads the **prefix** (everything typed so far on the current line)
* Decoder autoregressively generates the **rest of the line** until `<EOS>`

Both models share the same `CodeTokenizer` vocabulary.

---

## Quick-start

### 1. Install dependencies
```bash
pip install torch numpy matplotlib
```

### 2. Prepare data
Put your GitHub .py files anywhere under `raw_data/`, then run:
```bash
python prepare_data.py \
    --raw_dir  raw_data \
    --out_dir  data \
    --min_lines 10 \
    --syntax_check          # optional: filter files that don't parse
```
This will:
* Deduplicate by MD5 hash
* Filter low-quality files
* Split into train / val / test (90 / 5 / 5 by default)
* Save a dataset statistics PNG to `data/dataset_stats.png`

### 3. Train
```bash
python train.py \
    --data_dir data/train \
    --epochs   30 \
    --batch    64 \
    --lr       3e-4 \
    --d_model  256 \
    --n_layers 4 \
    --n_heads  8
```

Training flags:
| Flag | Default | Description |
|---|---|---|
| `--epochs` | 20 | training epochs |
| `--batch` | 32 | batch size |
| `--lr` | 3e-4 | peak learning rate |
| `--ctx` | 128 | token model context window |
| `--d_model` | 256 | transformer hidden dim |
| `--n_layers` | 4 | encoder/decoder depth |
| `--n_heads` | 8 | attention heads |
| `--vocab_size` | 8000 | tokenizer vocabulary |
| `--skip_token` | — | skip token model training |
| `--skip_line` | — | skip line model training |
| `--val_split` | 0.1 | fraction of data held out for validation |

Metric plots are saved to `plots/` every ~20% of epochs **and** at the end.
Best checkpoints (by val loss, top-3) are saved to `checkpoints/`.

### 4. Evaluate
```bash
python evaluate.py \
    --data_dir  data/val \
    --ckpt_dir  checkpoints \
    --tokenizer tokenizer.json \
    --out_dir   eval_results
```

Metrics reported:

**Token model**
* Top-1 / Top-5 accuracy
* Mean Reciprocal Rank (MRR)
* Perplexity
* Latency (ms/sample)

**Line model**
* Exact-match@1
* Prefix-20 match
* BLEU-4
* chrF
* Latency (ms/sample)

Results are saved to `eval_results/eval_results.json` and plotted in
`eval_results/eval_dashboard.png`.

### 5. Interactive hand-test
After training, an interactive REPL starts automatically. You can also
launch it manually:
```bash
python train.py --test
```

REPL commands:
```
>> :token def fibonacci(n     ← next-word completion
>> :line  for i in range(     ← full-line completion
>> :temp 0.7                  ← set sampling temperature
>> :k 40                      ← set top-k
>> :quit
```
Completions are printed in colour:
* 🔵 token completions in green
* 🟡 line  completions in yellow

---

## Metric dashboards

Each saved PNG contains a 2×3 grid:

| Panel | Description |
|---|---|
| Train / Val Loss | Cross-entropy loss curves |
| Train / Val Perplexity | Log-scale perplexity |
| Top-1 Token Accuracy | Fraction of correct next-token predictions |
| Learning Rate | Cosine-annealed LR schedule |
| Gradient Norm | Clipped gradient norm per epoch |
| Generalisation Gap | Val − Train loss (overfitting indicator) |

---

## Hyperparameter tuning tips

| Goal | Suggestion |
|---|---|
| Better quality, more VRAM | Increase `--d_model` to 512, `--n_layers` to 6 |
| Faster training | Reduce `--ctx` to 64, `--batch` up to 128 |
| Reduce overfitting | Lower `--lr`, add `--val_split 0.15` |
| Better line completion | Increase `--epochs` for the line model |

---

## File overview

| File | Purpose |
|---|---|
| `prepare_data.py` | Filter, deduplicate, and split raw .py files |
| `train.py` | Tokenizer, both model classes, training loops, REPL |
| `evaluate.py` | Offline evaluation with BLEU, chrF, accuracy, MRR |
| `requirements.txt` | `torch`, `numpy`, `matplotlib` |
