# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import torch
from torch import Tensor


_PAD_SEGMENT_ID = -1


def pack(
    token_seqs: list[dict[str, list[int | float]]],
    sample_attrs: list[dict[str, int | float]],
    max_seq_length: int,
    num_rows: int,
    pad_values: dict[str, int | float],
    balance_attn_cost: bool = False,
) -> dict[str, Any]:
    """Pack variable-length samples into [num_rows, max_seq_length] tensors.

    Shared between SFT (as a DataLoader collate function) and RL (called
    directly). First-fit row assignment with rotating start; degenerates to
    greedy sequential fill when num_rows=1.

    Args:
        token_seqs: Per-sample per-token fields. All fields within a sample
            must have the same length. Each field is concat + padded to
            max_seq_length per row. Example keys: "input_ids", "label_ids",
            "ref_logprobs", "loss_mask".
        sample_attrs: Per-sample scalars collected per row without padding.
            Example keys: "advantage", "reward".
        max_seq_length: Target row length. Rows are padded to this length.
        num_rows: Number of rows in the output. SFT: batch_size. RL: microbatch_size.
        pad_values: Pad value for each token_seqs key.
        balance_attn_cost: If True, assign samples to the row with the
            smallest cumulative attention cost (sum of seqlen²) instead of
            first-fit. Useful when sequence lengths vary widely.

    Returns:
        Dict with:
            "token_seqs": dict[str, Tensor] — all [num_rows, max_seq_length]
            "segment_ids": Tensor [num_rows, max_seq_length]
            "sample_attrs": dict[str, list[list[float]]]
    """
    if not token_seqs:
        raise ValueError("token_seqs must not be empty")

    field_keys = list(token_seqs[0].keys())
    first_key = field_keys[0]
    sample_lengths = [len(ts[first_key]) for ts in token_seqs]

    attr_keys = list(sample_attrs[0].keys()) if sample_attrs and sample_attrs[0] else []

    # --- Assign each sample to a row via first-fit ---
    row_assignments: list[list[int]] = [[] for _ in range(num_rows)]
    row_lengths = [0] * num_rows
    row_attn_costs = [0] * num_rows
    rotate_idx = 0

    for i, length in enumerate(sample_lengths):
        if length > max_seq_length:
            continue

        row_idx = _find_row(
            length,
            num_rows,
            max_seq_length,
            row_lengths,
            row_attn_costs,
            rotate_idx,
            balance_attn_cost,
        )

        if row_idx < 0:
            continue

        row_assignments[row_idx].append(i)
        row_lengths[row_idx] += length
        if not balance_attn_cost:
            rotate_idx = row_idx

    # --- Pack all rows ---
    all_token_seqs: dict[str, list[Tensor]] = {k: [] for k in field_keys}
    all_segment_ids: list[Tensor] = []
    all_sample_attrs: dict[str, list[list[float]]] = {k: [] for k in attr_keys}

    for row_indices in row_assignments:
        row_data = _pack_row(
            row_indices,
            token_seqs,
            sample_attrs,
            field_keys,
            attr_keys,
            first_key,
            max_seq_length,
            pad_values,
        )

        for key in field_keys:
            all_token_seqs[key].append(row_data["token_seqs"][key])
        all_segment_ids.append(row_data["segment_ids"])
        for key in attr_keys:
            all_sample_attrs[key].append(row_data["sample_attrs"][key])

    return {
        "token_seqs": {k: torch.stack(v) for k, v in all_token_seqs.items()},
        "segment_ids": torch.stack(all_segment_ids),
        "sample_attrs": all_sample_attrs,
    }


def _find_row(
    length: int,
    num_rows: int,
    max_seq_length: int,
    row_lengths: list[int],
    row_attn_costs: list[int],
    rotate_idx: int,
    balance_attn_cost: bool,
) -> int:
    """Find the best row for a sample of given length.

    Default: first-fit with rotating start.
    balance_attn_cost: min-cost (smallest sum of seqlen²) among eligible rows.
    """
    if balance_attn_cost:
        best_idx = -1
        best_cost = float("inf")
        for i in range(num_rows):
            if row_lengths[i] + length <= max_seq_length:
                if row_attn_costs[i] < best_cost:
                    best_cost = row_attn_costs[i]
                    best_idx = i
        if best_idx >= 0:
            row_attn_costs[best_idx] += length * length
        return best_idx

    for offset in range(num_rows):
        idx = (rotate_idx + offset) % num_rows
        if row_lengths[idx] + length <= max_seq_length:
            return idx
    return -1


def _pack_row(
    row_indices: list[int],
    token_seqs: list[dict[str, list[int | float]]],
    sample_attrs: list[dict[str, int | float]],
    field_keys: list[str],
    attr_keys: list[str],
    first_key: str,
    max_seq_length: int,
    pad_values: dict[str, int | float],
) -> dict:
    """Pack samples assigned to one row: concat, assign segment_ids, pad."""
    row_field_data: dict[str, list] = {k: [] for k in field_keys}
    row_segment_ids: list[int] = []
    row_attr_data: dict[str, list[float]] = {k: [] for k in attr_keys}

    for seg_id, sample_idx in enumerate(row_indices):
        ts = token_seqs[sample_idx]
        n = len(ts[first_key])

        for key in field_keys:
            row_field_data[key].extend(ts[key])

        row_segment_ids.extend([seg_id] * n)

        if sample_attrs:
            sa = sample_attrs[sample_idx]
            for key in attr_keys:
                row_attr_data[key].append(sa[key])

    # Pad to max_seq_length
    total_tokens = len(row_segment_ids)
    pad_len = max_seq_length - total_tokens
    if pad_len > 0:
        for key in field_keys:
            row_field_data[key].extend([pad_values[key]] * pad_len)
        row_segment_ids.extend([_PAD_SEGMENT_ID] * pad_len)

    return {
        "token_seqs": {
            k: torch.tensor(
                v, dtype=torch.long if k.endswith("_ids") else torch.float32
            )
            for k, v in row_field_data.items()
        },
        "segment_ids": torch.tensor(row_segment_ids, dtype=torch.long),
        "sample_attrs": row_attr_data,
    }
