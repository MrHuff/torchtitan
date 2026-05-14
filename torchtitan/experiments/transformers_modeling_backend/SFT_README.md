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

#### `model.py` — HFFlexAttention + _flex_torchtitan_attention_forward

**`HFFlexAttention`** subclasses TorchTitan's `FlexAttention` kernel module.
It accepts Q/K/V in native TorchTitan layout `(batch, seq, heads, dim)`,
transposes to `(batch, heads, seq, dim)` for the `flex_attention` kernel, and
transposes back. Each HF attention layer gets one registered as a submodule
via `register_module("_flex_kernel", ...)` during model init. This makes
`apply_cp_to_forward`'s isinstance check pass, so CP wrapping applies
automatically — no globals needed.

**`_flex_torchtitan_attention_forward`** is the function registered via
`AttentionInterface`. It transposes Q/K/V from HF layout to native layout,
calls the `HFFlexAttention` module (which may have been wrapped by
`apply_cp_to_forward` for K/V all-gather), and transposes back:

```python
def _flex_torchtitan_attention_forward(module, query, key, value, ...):
    flex_module = module._flex_kernel           # attached HFFlexAttention
    q = query.transpose(1, 2)                   # HF → native layout
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)
    out = flex_module(q, k, v, attention_masks=block_mask, scale=scaling)
    return out.transpose(1, 2), None            # native → HF layout
```

**Why not use HF's `flex_attention_forward`?** Two reasons:

1. HF's version passes `return_lse=True` to PyTorch's `flex_attention`, which
   triggers `warnings.warn` inside PyTorch. `torch.compile`'s dynamo can't
   trace `warnings.warn`, causing a graph break and compilation failure.

2. By using a custom name `"flex_torchtitan"` (not registered in HF's mask
   registry), HF's `create_causal_mask` skips building its own mask. This is
   critical for TP + Sequence Parallel — HF would build the mask at the wrong
   sequence length (after SP has split the input).

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

**`HFFlexAttention` class:** Subclasses TorchTitan's `FlexAttention` kernel
module. This is what makes CP work — `apply_cp_to_forward` checks
`isinstance(module, FlexAttention)`, which passes for our subclass. The module
accepts Q/K/V in native TorchTitan layout `(batch, seq, heads, dim)`,
transposes to `(batch, heads, seq, dim)` for the `flex_attention` kernel, and
transposes back. `_flex_torchtitan_attention_forward` handles the HF↔native
layout conversion before/after calling this module.

Each HF attention layer gets an `HFFlexAttention` instance registered as a
submodule during model init:

```python
layer.self_attn.register_module(
    "_flex_kernel", HFFlexAttention(config=HFFlexAttention.Config())
)
```

Using `register_module` (not plain attribute assignment) makes it a proper
`nn.Module` submodule — visible to `named_modules()`, moved by `.to(device)`,
included in `state_dict()`.

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

#### `parallelize.py` — CP wrapping

Added `apply_cp_to_forward` call when CP is enabled. Collects the
`HFFlexAttention` modules from each HF attention layer and passes them to
`apply_cp_to_forward`, which wraps each module's `forward` to all-gather K/V
before attention:

```python
if parallel_dims.cp_enabled:
    model.set_cp_mesh(parallel_dims.get_mesh("cp"))
    flex_modules = []
    for layer in model.layers.values():
        if hasattr(layer.self_attn, "_flex_kernel"):
            flex_modules.append(layer.self_attn._flex_kernel)
    if flex_modules:
        apply_cp_to_forward(flex_modules, parallel_dims.get_mesh("cp"))
```

This follows the same pattern as native model parallelization (e.g.,
`llama3/parallelize.py`) where `apply_cp_to_forward` is called with the inner
attention modules before `parallelize_module()`.

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

| Parallelism | Status | Changes for SFT | Why |
|---|---|---|---|
| FSDP | Works | None | Operates on weights, not masks |
| FSDP + TP | Works | BlockMask built before SP split (SFTTrainer) | SP changes input shape |
| FSDP + TP + PP | Works | None | BlockMask in extra_kwargs, forwarded automatically |
| FSDP + compile | Works | Custom attention fn + cache limit (model.py) | HF code breaks dynamo |
| FSDP + TP + compile | Works | cache_size_limit=64 (model.py) | HF code complexity |
| FSDP + AC (selective/full) | Works | None | Operates on activations, not masks |
| FSDP + CP | Works | K/V all-gather + ptrr (model.py + trainer.py) | HF doesn't use native FlexAttention module |

### Parallelisms That Work Out of the Box

**FSDP** shards model weights across GPUs. Each GPU stores a fraction of each
weight matrix and all-gathers the full weight before each layer's forward pass.
This operates entirely on weights — it doesn't interact with the attention mask
or data pipeline. SFT's BlockMask passes through FSDP untouched.

**PP (Pipeline Parallel)** splits layers across GPUs. GPU 0 runs layers 0-15,
GPU 1 runs layers 16-31. The BlockMask is in `extra_kwargs`, which PP
automatically forwards to all stages. Every stage's attention layers receive
the same BlockMask. No changes needed — we just put the mask in `extra_kwargs`
(forwarded to all stages) instead of `extra_inputs` (first stage only).

**AC (Activation Checkpointing)** discards intermediate activations during
forward to save memory, then recomputes them during backward. This operates on
layer activations, not on the attention mask. The BlockMask is an input to
each layer, not a saved activation, so AC has no interaction with SFT.

### Why TP Required Special Handling

**TP (Tensor Parallel)** splits weight matrices across GPUs — each GPU computes
a slice of Q, K, V, and MLP outputs. TP itself needed no changes for SFT.

The issue is **Sequence Parallel (SP)**, which is enabled alongside TP. SP
splits the input along the sequence dimension for non-attention operations
(LayerNorm, Dropout). With seq_len=2048 and TP=2, each GPU gets 1024 tokens.

If HF builds the mask inside `forward()` (as it does with
`attn_implementation="flex_attention"`), it sees the split sequence length
(1024) and builds a mask for the wrong size. Inside the attention layer, Q/K/V
are all-gathered back to 2048, but the mask says 1024 → crash.

Our approach builds the mask in `post_dataloading_process` (before SP splits
the input) so it always sees the full sequence length (2048). This is why we
use the custom name `"flex_torchtitan"` — it prevents HF from building its own
mask at the wrong point.

### Why CP Required Special Handling

#### What CP does

CP splits the sequence across GPUs. With 2 GPUs and seq_len=2048, each GPU
gets 1024 tokens. But attention needs every query to potentially see every key —
token 500 (GPU 0) might need to attend to token 1500 (GPU 1).

CP solves this by keeping Q local but **all-gathering K,V from all GPUs**:

```
GPU 0:
  Q  = local [tokens 0-1023]
  K,V = all_gather → [tokens 0-2047]    ← full sequence
  attention(Q=1024, KV=2048)             → output for tokens 0-1023

GPU 1:
  Q  = local [tokens 1024-2047]
  K,V = all_gather → [tokens 0-2047]    ← full sequence
  attention(Q=1024, KV=2048)             → output for tokens 1024-2047
```

The BlockMask is split along Q only (each GPU's queries), but KV stays full:

```
Full BlockMask (2048 × 2048):        GPU 0's BlockMask (1024 × 2048):
     KV: 0────────────2048                KV: 0────────────2048
Q: 0    [ConvA |      ]            Q: 0    [ConvA |      ]
        [      |      ]                    [      |      ]
Q:1024  [      | ConvB ]            GPU 1's BlockMask (1024 × 2048):
        [      |      ]                 KV: 0────────────2048
Q:2048                              Q:1024  [      | ConvB ]
                                           [      |      ]
```

#### How CP wrapping works

`apply_cp_to_forward()` wraps each `FlexAttention` module's forward to
all-gather K,V before attention:

```python
if isinstance(first, FlexAttention):       # checks for FlexAttention module
    def cp_forward(q, k, v):
        global_k, global_v = flex_cp_allgather(k, v, dim=1)  # dim 1 = seq in native layout
        return orig_fn(q, global_k, global_v)
    mod.forward = cp_forward
```

HF models don't have TorchTitan's `FlexAttention` module — they have their own
attention classes (`LlamaAttention`, `Qwen3Attention`). To make
`apply_cp_to_forward` work, we:

1. **Subclass `FlexAttention`** as `HFFlexAttention` — passes the isinstance
   check
2. **Register it as a submodule** on each HF attention layer via
   `register_module("_flex_kernel", ...)`
3. **Collect these modules** in `parallelize.py` and pass them to
   `apply_cp_to_forward`

The layout difference is handled by transposes:
- `_flex_torchtitan_attention_forward` transposes Q/K/V from HF layout
  `(batch, heads, seq, dim)` to native layout `(batch, seq, heads, dim)`
  before calling the module
- `apply_cp_to_forward`'s all-gather runs on dim 1 (sequence in native layout)
  — correct
- `HFFlexAttention.forward` transposes back to `(batch, heads, seq, dim)` for
  the `flex_attention` kernel, then transposes the output back to native layout
- `_flex_torchtitan_attention_forward` transposes the final output back to HF
  layout

We also switch the load balancer from `"headtail"` to `"ptrr"` for
block-causal masks in `SFTTrainer.post_dataloading_process`.

#### The full CP flow

```
1. SFTTrainer.post_dataloading_process:
   - Build BlockMask for full seq_len (2048 × 2048)
   - prepare_context_parallel_input with ptrr load balancer:
     - Shards input:    [batch, 2048] → [batch, 1024] per rank
     - Shards positions: [batch, 2048] → [batch, 1024] per rank
     - Shards BlockMask along Q dim: (2048, 2048) → (1024, 2048) per rank

2. Model forward:
   - Each rank processes its local chunk through embeddings + layers
   - Q, K, V are all (batch, heads, 1024, dim) in HF layout (local chunk)

3. _flex_torchtitan_attention_forward:
   - Transposes Q/K/V from HF layout to native layout
   - Calls HFFlexAttention module (which apply_cp_to_forward has wrapped)
   - CP wrapper all-gathers K/V along dim 1 (seq in native layout):
     K: (batch, 1024, heads, dim) → (batch, 2048, heads, dim)
     V: (batch, 1024, heads, dim) → (batch, 2048, heads, dim)
   - HFFlexAttention transposes to (batch, heads, seq, dim) for kernel
   - flex_attention(Q=1024, K=2048, V=2048, mask=(1024, 2048)) → works ✓
   - Transposes back through native layout to HF layout
```

### Why compile Required Special Handling

`torch.compile` traces Python code into a computation graph, then generates
fused GPU kernels. Two issues arose with SFT:

**Problem 1: `warnings.warn` graph break.** HF's `flex_attention_forward`
passes `return_lse=True` to PyTorch's `flex_attention`, which internally calls
`warnings.warn` about a deprecation. Dynamo can't trace `warnings.warn` — it's
a Python builtin that dynamo marks as "skipped." This causes a graph break,
which fails with `fullgraph=True`. Our `_flex_torchtitan_attention_forward`
calls `flex_attention` directly without `return_lse`, avoiding the warning.

**Problem 2: Recompilation limit with TP.** When `torch.compile` traces a
function, it records "guards" — conditions under which the compiled graph is
valid (input shapes, types, dtypes). If a guard fails on the next call, dynamo
recompiles. HF model code has more Python conditionals, wrappers, and dispatch
logic than native TorchTitan, creating more guards. During the first few
training steps, TP's async collectives produce different tensor types
(`AsyncCollectiveTensor` vs `Tensor`) as communication patterns stabilize.
More guards × more type variations = more recompilations. The default limit of
8 was too low; we set `cache_size_limit=64` to let the guards stabilize. After
3-5 steps, no more recompilation occurs — the cache has all the graphs it
needs.

Note: pretraining with TP + compile works at the default limit because native
TorchTitan model code is simpler (fewer guards). The higher limit is specific
to HF models' more complex Python code paths.

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
