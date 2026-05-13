# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Iterable, Iterator

import torch


_PAD_SEGMENT_ID = -1


def pack(
    samples: Iterable[dict[str, list]],
    max_seq_length: int,
    pad_values: dict[str, int | float],
) -> Iterator[dict[str, torch.Tensor]]:
    """Greedy-pack variable-length samples into [1, max_seq_length] sequences.

    Takes an iterable of samples, appends each to a buffer. When the next
    sample doesn't fit, pads and yields the current buffer, then starts a
    new one with that sample. Yields remaining buffer at the end.

    This is a generator — it yields one packed dict each time the buffer
    is full. Same streaming behavior as SFT's _iter_greedy_packed.

    Args:
        samples: Iterable of dicts. Each dict maps field names to lists of
            the same length. E.g. {"input_ids": [1,2,3], "label_ids": [2,3,4]}.
        max_seq_length: Maximum tokens per packed sequence.
        pad_values: Pad value for each field key.

    Yields:
        Dict with field tensors [1, max_seq_length] and
        "segment_ids" tensor [1, max_seq_length].
    """
    field_keys: list[str] | None = None
    buffer: dict[str, list] = {}
    buffer_segment_ids: list[int] = []
    buffer_length = 0
    segment_id = 0

    def _flush() -> dict[str, torch.Tensor]:
        nonlocal buffer, buffer_segment_ids, buffer_length, segment_id
        assert field_keys is not None
        pad_length = max_seq_length - buffer_length
        if pad_length > 0:
            for key in field_keys:
                buffer[key].extend([pad_values[key]] * pad_length)
            buffer_segment_ids.extend([_PAD_SEGMENT_ID] * pad_length)

        result = {
            key: torch.tensor(
                values, dtype=torch.long if key.endswith("_ids") else torch.float32
            ).unsqueeze(0)
            for key, values in buffer.items()
        }
        result["segment_ids"] = torch.tensor(
            buffer_segment_ids, dtype=torch.long
        ).unsqueeze(0)

        buffer = {key: [] for key in field_keys}
        buffer_segment_ids = []
        buffer_length = 0
        segment_id = 0
        return result

    for sample in samples:
        if field_keys is None:
            field_keys = list(sample.keys())
            buffer = {key: [] for key in field_keys}

        first_key = field_keys[0]
        sample_length = len(sample[first_key])

        if sample_length > max_seq_length:
            continue

        if buffer_length > 0 and buffer_length + sample_length > max_seq_length:
            yield _flush()

        for key in field_keys:
            buffer[key].extend(sample[key])
        buffer_segment_ids.extend([segment_id] * sample_length)
        buffer_length += sample_length
        segment_id += 1

    if buffer_length > 0:
        yield _flush()
