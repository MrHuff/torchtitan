# SFT for HuggingFace Models in TorchTitan

This document explains the SFT (Supervised Fine-Tuning) implementation for the
`transformers_modeling_backend` experiment. It covers what was built, how each
component works, and why certain design decisions were made.

## Background: What is SFT and Why is it Different from Pretraining?

**Pretraining** trains a model on raw text. Every token predicts the next token.
The attention pattern is simple causal — each token sees all previous tokens:

```
Token:  0  1  2  3  4  5  6  7
    0 [ ✓  .  .  .  .  .  .  . ]
    1 [ ✓  ✓  .  .  .  .  .  . ]
    2 [ ✓  ✓  ✓  .  .  .  .  . ]
    ...
```

**SFT** trains on conversations (user question → assistant answer). Two key
differences:

1. **Prompt masking**: Only the assistant's response tokens contribute to the
   loss. The user's prompt tokens are masked with `IGNORE_INDEX=-100` so the
   loss function skips them.

2. **Sequence packing**: Conversations are short (50-200 tokens), but GPUs are
   efficient with long sequences (2048 tokens). Multiple conversations are
   packed into one sequence. But tokens from different conversations must NOT
   attend to each other.

This creates a **block-causal** attention pattern:

```
[Conv A: "What is 2+3?" → "5"][Conv B: "Capital of France?" → "Paris"]

Token:  0  1  2  3  4  5  6  7
    0 [ ✓  .  .  .  .  .  .  . ]   Conv A
    1 [ ✓  ✓  .  .  .  .  .  . ]   Conv A
    2 [ ✓  ✓  ✓  .  .  .  .  . ]   Conv A
    3 [ ✓  ✓  ✓  ✓  .  .  .  . ]   Conv A
    4 [ .  .  .  .  ✓  .  .  . ]   Conv B ← blocked from Conv A
    5 [ .  .  .  .  ✓  ✓  .  . ]   Conv B
    6 [ .  .  .  .  ✓  ✓  ✓  . ]   Conv B
    7 [ .  .  .  .  ✓  ✓  ✓  ✓ ]   Conv B
```

## Why FlexAttention?

SDPA (Scaled Dot-Product Attention) has `is_causal=True` which handles the
simple causal pattern. But it can't express block-causal. If you pass a custom
mask tensor, SDPA falls back to a slow "math" backend (no Flash Attention).

FlexAttention lets you define the mask as a function:

```python
def mask(b, h, q_idx, kv_idx):
    return doc_ids[q_idx] == doc_ids[kv_idx] and q_idx >= kv_idx
```

This function gets compiled into the GPU attention kernel via `torch.compile`.
The kernel:
- Skips entire tiles of the attention matrix that are fully masked (e.g., all
  of Conv A queries vs Conv B keys — zero compute)
- Applies the mask check using fast GPU registers (not a memory read from a
  mask tensor)
- Runs at Flash Attention speed with zero mask memory overhead

The pretraining path doesn't need FlexAttention because `is_causal=True`
already handles simple causal masking efficiently. FlexAttention is only needed
for the custom block-causal pattern in SFT.

## Architecture Overview

```
ChatDataLoader (core TorchTitan, from PR #2556)
  → tokenizes conversations, masks prompt labels with -100
  → packs multiple conversations per sequence
  → positions reset to 0 at each conversation boundary
  → yields {input, positions}, labels
        │
        ▼
SFTTrainer.post_dataloading_process()           ← BEFORE parallelism
  → detects attn_mask_type="block_causal"
  → builds BlockMask from positions
  → if CP enabled: shards inputs + BlockMask
  → passes BlockMask + positions in extra_kwargs
        │
        ▼
HFTransformerModel.forward()
  → passes position_ids to HF model (for RoPE)
  → passes BlockMask as attention_mask to HF model
        │
        ▼
HF model's attention layer
  → dispatches to _flex_torchtitan_attention_forward
  → if CP: all-gathers K/V from other ranks
  → calls flex_attention(Q, K, V, block_mask=BlockMask)
        │
        ▼
Loss (core TorchTitan)
  → cross_entropy with ignore_index=-100 skips prompt tokens
  → normalized by global_valid_tokens across DP ranks
```

## Files Changed / Added

All changes are within `torchtitan/experiments/transformers_modeling_backend/`.
No core TorchTitan code was modified.

### New Files

#### `trainer.py` — SFTTrainer

**What it does:** Builds a FlexAttention `BlockMask` from per-document positions
before any parallelism sharding happens.

**Why it exists:** The base `Trainer.post_dataloading_process()` checks
`isinstance(self.model_config, Decoder.Config)` to decide whether to build
attention masks. HF models fail this check, so masks are never built. We
override `post_dataloading_process` to build the BlockMask for HF models.

**Why the mask must be built here (before parallelism):** With Tensor Parallel +
Sequence Parallel, the input gets split along the sequence dimension AFTER
`post_dataloading_process`. If we built the mask later (inside the model's
`forward()`), we'd see the split sequence length (e.g., 1024 instead of 2048)
and build a mask for the wrong size.

**How it builds the mask:**

```python
mask_mod = and_masks(
    get_causal_mask_mod(),           # q_idx >= kv_idx (causal)
    get_document_mask_mod(positions)  # same document check
)
extra_kwargs["attention_masks"] = create_attention_mask(
    mask_mod, B, None, seq_len, seq_len
)
```

These are the exact same functions native TorchTitan uses in
`Decoder._get_flex_attention_masks()`. The document mask detects conversation
boundaries from positions resetting to 0.

**Context Parallel handling:** When CP is enabled, the BlockMask and inputs are
sharded via `prepare_context_parallel_input` with the `"ptrr"` load balancer
(required for FlexAttention with CP).

#### `state_dict_adapter.py` — HFTransformerStateDictAdapter

**What it does:** Converts between TorchTitan state dict keys and HF safetensors
keys for loading/saving pretrained weights.

**How it works:** Since `HFTransformerModel` wraps an HF `ForCausalLM` as
`self.model`, the only difference is a `model.` prefix:

```
TorchTitan key:  model.model.layers.0.self_attn.q_proj.weight
HF safetensors:  model.layers.0.self_attn.q_proj.weight
                 ↑ strip this prefix
```

- `to_hf()`: strips `model.` prefix
- `from_hf()`: adds `model.` prefix

No weight reshaping or renaming needed — the model IS an HF model.

**Weight tying:** Some models (Qwen2.5, Granite) set `tie_word_embeddings=True`
and omit `lm_head.weight` from their safetensors file (it shares storage with
`embed_tokens.weight`). The adapter handles this:

- `to_hf()`: removes `lm_head.weight` from the dict before loading so DCP
  doesn't fail on a missing key
- `from_hf()`: copies `embed_tokens.weight` to `lm_head.weight` after loading

#### `tokenizer.py` — HFBackendTokenizer

**What it does:** Passes special tokens to chat templates and fixes an EOS
token edge case.

**Problem 1 — Chat templates need special tokens:** Some HF models' Jinja chat
templates reference `bos_token` and `eos_token` (e.g., Seed-Coder). The base
`HuggingFaceTokenizer.apply_chat_template()` only passes `messages` to the
template, not special tokens. Our subclass injects them:

```python
def apply_chat_template(self, messages, **kwargs):
    kwargs.setdefault("bos_token", self.bos_token or "")
    kwargs.setdefault("eos_token", self.eos_token or "")
    return super().apply_chat_template(messages, **kwargs)
```

**Problem 2 — Shared BOS/EOS token:** Granite uses `<|end_of_text|>` for BOTH
`bos_token` and `eos_token`. The base tokenizer has an `if/elif` that sets
BOS but skips EOS when they're the same string. The `ChatDataset` then crashes
with "Tokenizer does not have an eos_id set." Our `__init__` detects this and
sets EOS explicitly.

### Modified Files

#### `model.py` — _flex_torchtitan_attention_forward + changes

**Custom attention forward function:**

```python
def _flex_torchtitan_attention_forward(
    module, query, key, value, attention_mask, **kwargs
):
    block_mask = attention_mask if isinstance(attention_mask, BlockMask) else None
    out = flex_attention(query, key, value, block_mask=block_mask,
                         scale=scaling, enable_gqa=True)
    return out, None
```

**Why not use HF's `flex_attention_forward`?** Two reasons:

1. HF's version passes `return_lse=True` to PyTorch's `flex_attention`, which
   triggers `warnings.warn` inside PyTorch. `torch.compile`'s dynamo can't
   trace `warnings.warn`, causing a graph break and compilation failure. Our
   function avoids `return_lse` entirely.

2. By using a custom name `"flex_torchtitan"` (not registered in HF's mask
   registry), HF's `create_causal_mask` skips building its own mask. This is
   critical for TP + Sequence Parallel — HF would build the mask at the wrong
   sequence length (after SP has split the input).

**Context Parallel support:** When CP is active, K/V are all-gathered before
attention:

```python
if _cp_mesh is not None:
    key, value = flex_cp_allgather(key, value, 2, pg_name)
    #                                         ^ dim 2 = sequence
    # HF layout: (batch, heads, seq, dim) — dim 2 is sequence
    # Native TorchTitan layout: (batch, seq, heads, dim) — dim 1 is sequence
```

**Attention registration in `_configure_hf_attention`:**

```python
if attn_implementation == "flex_torchtitan":
    AttentionInterface._global_mapping[attn_implementation] = _flex_torchtitan_attention_forward
elif attn_implementation not in AttentionInterface._global_mapping:
    AttentionInterface._global_mapping[attn_implementation] = sdpa_attention_forward
```

- `"flex_torchtitan"`: registers our FlexAttention forward (SFT path)
- `"sdpa_torchtitan"`: registers SDPA with `is_causal=True` (pretraining path,
  unchanged)

**`forward()` changes:** Accepts optional `positions` and `attention_masks`
kwargs. When provided, passes them to the HF model. Falls back to sequential
`arange` positions when `None` (pretraining path unchanged).

**Other `model.py` changes:**

- `torch._dynamo.config.cache_size_limit = 64`: HF model code has more Python
  complexity (conditionals, wrappers, dispatch) than native TorchTitan, causing
  more dynamo guard variations during TP warmup. The default limit of 8 is too
  low; 64 gives room to stabilize.

- `attention_dropout = 0.0`: FlexAttention doesn't support dropout. Models like
  Seed-Coder have `attention_dropout=0.1` by default.

- `intermediate_size` / `head_dim` guarding: Only recalculate these when model
  dimensions are explicitly overridden (debugmodel). When using the `full`
  flavor, trust the HF config's values so pretrained weight shapes match.

#### `__init__.py` — Flavors and model_registry

Added `sft_debugmodel` and `sft_full` flavors:

```python
"sft_debugmodel": HFTransformerModel.Config(
    titan_dense_config=TitanDenseModelConfig(
        dim=256, n_layers=2, n_heads=16, n_kv_heads=16,
        attn_mask_type="block_causal",
    ),
    attn_implementation="flex_torchtitan",
),
"sft_full": HFTransformerModel.Config(
    titan_dense_config=TitanDenseModelConfig(
        attn_mask_type="block_causal",
    ),
    attn_implementation="flex_torchtitan",
),
```

- `attn_mask_type="block_causal"`: tells `SFTTrainer` to build a BlockMask
- `attn_implementation="flex_torchtitan"`: uses our FlexAttention forward,
  bypasses HF's internal mask creation

`TitanDenseModelConfig` defaults changed to `None` for `dim`, `n_layers`,
`n_heads`, `n_kv_heads` so `full` flavors inherit architecture from the HF
config (via `AutoConfig.from_pretrained`) instead of overriding it.

Wired `HFTransformerStateDictAdapter` into `model_registry()` for weight
loading.

#### `configs.py` — TransformersBackendConfig

Overrides `build()` to instantiate `SFTTrainer` instead of `Trainer`.
`SFTTrainer` is a superset — for pretraining (`attn_mask_type="causal"`), it
skips mask building and behaves identically to the base `Trainer`.

#### `config_registry.py` — SFT config functions

Two new config functions:

- `transformers_modeling_backend_sft_debugmodel()`: Random init, tiny model, for
  CI testing. Uses `ChatDataLoader` with the test math Q&A dataset.
- `transformers_modeling_backend_sft_full()`: Loads pretrained HF weights via
  `initial_load_in_hf=True`. Default: Qwen3-0.6B.

Both use `HFBackendTokenizer` for chat template compatibility.

## How the Attention Path Works End-to-End

### Pretraining (unchanged)

```
attn_implementation = "sdpa_torchtitan"

1. "sdpa_torchtitan" NOT in HF's mask registry
   → HF skips mask creation entirely
2. SDPA runs with is_causal=True
   → Flash Attention kernel with hardcoded causal triangle
3. No mask tensor, no FlexAttention, maximum speed
```

### SFT

```
attn_implementation = "flex_torchtitan"

1. SFTTrainer builds BlockMask from positions (before parallelism)
   - get_causal_mask_mod(): q_idx >= kv_idx
   - get_document_mask_mod(positions): same conversation check
   - create_attention_mask(): compiles into BlockMask
   
2. "flex_torchtitan" NOT in HF's mask registry
   → HF skips internal mask creation (avoids TP/SP size mismatch)

3. BlockMask passed as attention_mask through HF model layers

4. HF attention layer dispatches to _flex_torchtitan_attention_forward
   - If CP: all-gathers K/V along sequence dim
   - Calls flex_attention(Q, K, V, block_mask=BlockMask)
   - Compiled kernel: skips masked blocks, fuses mask check in registers
```

## Parallelism Support

| Parallelism | Status | Notes |
|---|---|---|
| FSDP | Works | Standard data parallel sharding |
| FSDP + TP | Works | BlockMask built before SP split |
| FSDP + TP + PP | Works | BlockMask forwarded across PP stages |
| FSDP + compile | Works | Custom attention avoids warnings.warn graph break |
| FSDP + TP + compile | Works | cache_size_limit=64 for HF code complexity |
| FSDP + AC (selective/full) | Works | Standard activation checkpointing |
| FSDP + CP | Works | K/V all-gather in attention forward, ptrr load balancer |

### Why TP Required Special Handling

With TP + Sequence Parallel, the input is split along the sequence dimension
before the model's `forward()` runs. If HF builds the mask inside `forward()`
(as it does with `attn_implementation="flex_attention"`), it sees the split
sequence length (e.g., 1024 instead of 2048) and builds a mask for the wrong
size. Our approach builds the mask in `post_dataloading_process` (before the
split) so it always sees the full sequence length.

### Why CP Required Special Handling

CP shards the sequence across GPUs. Each GPU computes attention for its local
query chunk against the full K/V from all GPUs. Native TorchTitan wraps the
`FlexAttention` module's `forward` to all-gather K/V, but HF models don't use
TorchTitan's `FlexAttention` module. Our `_flex_torchtitan_attention_forward`
does the all-gather directly when CP is active. Note: HF uses layout
`(batch, heads, seq, dim)` so the gather is along dim 2 (sequence), not dim 1
as in native TorchTitan's `(batch, seq, heads, dim)` layout.

### Why compile Required Special Handling

HF's `flex_attention_forward` passes `return_lse=True` to PyTorch's
`flex_attention`, which triggers `warnings.warn` inside PyTorch. Dynamo can't
trace `warnings.warn`, causing a compilation failure. Our attention forward
function calls `flex_attention` directly without `return_lse`.

## Models Tested

| Model | Size | Status |
|---|---|---|
| Qwen3-0.6B | 596M | Works (pretrained weights) |
| Qwen2.5-0.5B-Instruct | 494M | Works (pretrained weights) |
| Llama-3.2-1B-Instruct | 1.2B | Works (pretrained weights) |
| Seed-Coder-8B-Instruct | 8.3B | Works (pretrained weights) |
| Granite-3.1-2B-Instruct | 2.5B | Works (pretrained weights) |
| Mistral-7B (debugmodel) | 18M | Works (random init) |

### Model Requirements for SFT

1. **Chat template**: The tokenizer must have a Jinja chat template (in
   `tokenizer_config.json` or a `chat_template.jinja` file). Base models
   usually don't have one — use the Instruct/Chat variant.

2. **Llama-like architecture**: The `_patch_hf_llama_like` init code assumes
   separate `q_proj`, `k_proj`, `v_proj` projections. Models with fused QKV
   (like Phi-3) will fail during weight initialization.

3. **EOS token**: `ChatDataset` requires `eos_id` for padding packed sequences.
   Our `HFBackendTokenizer` handles edge cases where BOS and EOS are the same
   string.

## How to Run

### Debug model (random init, no downloads)

```bash
NGPU=2 LOG_RANK=0 MODULE=transformers_modeling_backend \
  CONFIG=transformers_modeling_backend_sft_debugmodel ./run_train.sh
```

### Full model with pretrained weights

Download weights first:
```bash
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('Qwen/Qwen3-0.6B', \
  allow_patterns=['*.safetensors', '*.json', '*.jinja'], \
  local_dir='tests/assets/qwen3_0.6b')"
```

Then run:
```bash
rm -rf ./outputs/checkpoint && \
  NGPU=2 LOG_RANK=0 MODULE=transformers_modeling_backend \
  CONFIG=transformers_modeling_backend_sft_full ./run_train.sh
```

### Different model

```bash
rm -rf ./outputs/checkpoint && \
  NGPU=2 LOG_RANK=0 MODULE=transformers_modeling_backend \
  CONFIG=transformers_modeling_backend_sft_full ./run_train.sh \
  --hf_model meta-llama/Llama-3.2-1B-Instruct \
  --hf_assets_path ./tests/assets/llama3.2_1b_instruct
```

### With parallelism

```bash
# FSDP + TP
NGPU=4 LOG_RANK=0 MODULE=transformers_modeling_backend \
  CONFIG=transformers_modeling_backend_sft_debugmodel ./run_train.sh \
  --parallelism.tensor_parallel_degree 2 \
  --parallelism.data_parallel_shard_degree 2

# FSDP + compile
NGPU=2 LOG_RANK=0 MODULE=transformers_modeling_backend \
  CONFIG=transformers_modeling_backend_sft_debugmodel ./run_train.sh \
  --compile.enable

# FSDP + CP
NGPU=2 LOG_RANK=0 MODULE=transformers_modeling_backend \
  CONFIG=transformers_modeling_backend_sft_debugmodel ./run_train.sh \
  --parallelism.context_parallel_degree 2 \
  --parallelism.data_parallel_shard_degree 1
```

## Design Decisions

### Why `flex_torchtitan` instead of `flex_attention`?

Using HF's built-in `"flex_attention"` would let HF handle everything
automatically — it detects packed sequences from position_ids resets and builds
a BlockMask internally. We tried this (Option C) and it works for FSDP-only.

But it breaks with TP + Sequence Parallel because HF builds the mask inside
`forward()`, after SP has already split the input. The mask is built for the
wrong sequence length. By using a custom name that HF doesn't recognize in its
mask registry, we force HF to skip mask creation and build the mask ourselves
at the right point.

### Why SFTTrainer subclass instead of modifying core?

Per TorchTitan's experiment rules: "Don't modify core torchtitan code to
accommodate experiment-specific needs." The base `Trainer.post_dataloading_process`
checks `isinstance(self.model_config, Decoder.Config)` which won't match HF
models. A subclass is the clean way to override this.

### Why not use HF's `from_pretrained` for weight loading?

`HFTransformerModel` creates models on meta device, applies FSDP, then loads
weights via DCP (Distributed Checkpoint). HF's `from_pretrained` loads weights
eagerly on CPU then moves to GPU — incompatible with FSDP's meta device init.
The `state_dict_adapter` bridges TorchTitan's DCP loading with HF's safetensors
format.
