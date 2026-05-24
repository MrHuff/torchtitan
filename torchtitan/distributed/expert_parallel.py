# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import types

import torch
import torch.nn as nn
from torch.distributed._functional_collectives import (
    all_to_all_single,
    all_to_all_single_autograd,
    reduce_scatter_tensor_autograd,
)
from torch.distributed.tensor import (
    DeviceMesh,
    distribute_module,
    distribute_tensor,
    DTensor,
    Partial,
    Replicate,
    Shard,
)
from torch.distributed.tensor.parallel import ParallelStyle

from torchtitan.models.moe import utils as moe_utils
from torchtitan.models.moe.utils import _permute, _unpermute


_LBT_EP_DEBUG_SPLIT_COUNT = 0
_LBT_MOE_SCATTER_ADD_BF16 = None
_LBT_MOE_SCATTER_ADD_BF16_IMPORT_ATTEMPTED = False


def _lbt_env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def _lbt_ep_debug_split_limit() -> int:
    try:
        return int(os.environ.get("LBT_EP_DEBUG_SPLIT_LIMIT", "32"))
    except ValueError:
        return 32


def _lbt_ep_split_count_all_gather() -> bool:
    if "LBT_EP_SPLIT_COUNT_ALL_GATHER" in os.environ:
        return _lbt_env_flag("LBT_EP_SPLIT_COUNT_ALL_GATHER", False)
    return _lbt_ep_exact_equal_split_a2a()


def _lbt_split_stats(splits: list[int]) -> tuple[int, int, float]:
    if not splits:
        return 0, 0, 0.0
    total = sum(int(split) for split in splits)
    max_split = max(int(split) for split in splits)
    mean_split = total / len(splits) if total > 0 else 0.0
    ratio = max_split / mean_split if mean_split > 0 else 0.0
    return total, max_split, ratio


def _lbt_ep_a2a_mode(env_name: str = "LBT_EP_A2A_COMPRESS") -> str:
    value = os.environ.get(env_name, os.environ.get("LBT_EP_A2A_COMPRESS", "none"))
    value = value.lower()
    if value in ("", "0", "false", "no", "none", "off"):
        return "none"
    if value in ("1", "true", "yes", "on", "fp8", "e4m3", "fp8_e4m3"):
        return "fp8_e4m3"
    if value in ("e5m2", "fp8_e5m2"):
        return "fp8_e5m2"
    raise ValueError(
        f"Unsupported {env_name}={value}. Expected none, fp8_e4m3, or fp8_e5m2."
    )


def _lbt_ep_a2a_scale_mode() -> str:
    value = os.environ.get("LBT_EP_A2A_SCALE_MODE", "dynamic").lower()
    if value in ("static", "fixed"):
        return "static"
    if value in ("dynamic", "amax"):
        return "dynamic"
    raise ValueError(
        "Unsupported LBT_EP_A2A_SCALE_MODE="
        f"{value}. Expected dynamic or static."
    )


def _lbt_ep_a2a_compress_min_bytes() -> int:
    try:
        min_bytes = int(os.environ.get("LBT_EP_A2A_COMPRESS_MIN_BYTES", str(16 << 20)))
    except ValueError:
        min_bytes = 16 << 20
    return max(0, min_bytes)


def _lbt_should_compress_a2a_tensor(tensor: torch.Tensor) -> bool:
    min_bytes = _lbt_ep_a2a_compress_min_bytes()
    return min_bytes == 0 or tensor.numel() * tensor.element_size() >= min_bytes


def _lbt_ep_a2a_custom_autograd_below_threshold() -> bool:
    return _lbt_env_flag("LBT_EP_A2A_CUSTOM_AUTOGRAD_BELOW_THRESHOLD", True)


def _lbt_fp8_dtype(mode: str) -> torch.dtype:
    if mode == "fp8_e4m3":
        return torch.float8_e4m3fn
    if mode == "fp8_e5m2":
        return torch.float8_e5m2
    raise ValueError(f"Unsupported FP8 all-to-all mode: {mode}")


def _lbt_static_fp8_scale_value() -> float:
    try:
        scale = float(os.environ.get("LBT_EP_A2A_FP8_STATIC_SCALE", "1.0"))
    except ValueError:
        scale = 1.0
    if scale <= 0:
        scale = 1.0
    return scale


def _lbt_static_fp8_scale(device: torch.device) -> torch.Tensor:
    scale = _lbt_static_fp8_scale_value()
    return torch.tensor(scale, device=device, dtype=torch.float32)


def _lbt_gather_ep_scales(scale: torch.Tensor, group) -> torch.Tensor:
    world_size = torch.distributed.get_world_size(group)
    scales = torch.empty(world_size, device=scale.device, dtype=torch.float32)
    torch.distributed.all_gather_into_tensor(scales, scale.reshape(1), group=group)
    return scales


def _lbt_equal_splits(total: int, parts: int) -> list[int]:
    if parts <= 0:
        return []
    return [int(total) // int(parts) for _ in range(int(parts))]


def _lbt_ep_exact_equal_split_a2a() -> bool:
    mode = os.environ.get("LBT_MOE_EP_ROUTE_COVERAGE_MODE", "").strip().lower()
    degree = os.environ.get("LBT_MOE_EP_ROUTE_COVERAGE_DEGREE", "").strip()
    return (
        _lbt_env_flag("LBT_EP_EXACT_EQUAL_SPLIT_A2A")
        and _lbt_env_flag("LBT_MOE_EP_ROUTE_COVERAGE")
        and degree == "2"
        and mode in ("balanced", "balanced_fast", "fast_balanced")
    )


def _lbt_ep_scored_output_combine() -> bool:
    return _lbt_env_flag("LBT_EP_SCORED_OUTPUT_COMBINE")


def _lbt_ep_local_reduce_output_combine() -> bool:
    return _lbt_env_flag("LBT_EP_LOCAL_REDUCE_OUTPUT_COMBINE")


def _lbt_ep_pack_local_reduce_indices() -> bool:
    return _lbt_env_flag("LBT_EP_PACK_LOCAL_REDUCE_INDICES", True)


def _lbt_ep_local_reduce_collective() -> str:
    value = os.environ.get("LBT_EP_LOCAL_REDUCE_COLLECTIVE", "reduce_scatter")
    value = value.strip().lower()
    if value in ("rs", "reduce_scatter", "reducescatter"):
        return "reduce_scatter"
    return "all_to_all"


def _lbt_ep_local_reduce_a2a_bwd() -> bool:
    return _lbt_env_flag("LBT_EP_LOCAL_REDUCE_A2A_BWD")


def _lbt_ep_local_reduce_a2a_bwd_mode() -> str:
    value = os.environ.get("LBT_EP_LOCAL_REDUCE_A2A_BWD_MODE", "tokens")
    value = value.strip().lower()
    if value in ("route", "routes", "routed"):
        return "routes"
    return "tokens"


def _lbt_ep_score_dispatch_mode() -> str:
    value = os.environ.get("LBT_EP_SCORE_DISPATCH_MODE", "separate_fp32")
    value = value.strip().lower()
    if value in ("pack", "pack_bf16", "cat_bf16"):
        return "pack_bf16"
    if value in ("separate_bf16", "bf16"):
        return "separate_bf16"
    return "separate_fp32"


def _lbt_ep_local_reduce_index_dtype() -> str:
    value = os.environ.get("LBT_EP_LOCAL_REDUCE_INDEX_DTYPE", "int64")
    value = value.strip().lower()
    if value in ("auto", "int32", "i32", "32"):
        return "auto"
    return "int64"


def _lbt_ep_local_reduce_permuted() -> bool:
    return _lbt_env_flag("LBT_EP_LOCAL_REDUCE_PERMUTED")


def _lbt_ep_local_reduce_tk_scatter_add() -> bool:
    return _lbt_env_flag("LBT_EP_LOCAL_REDUCE_TK_SCATTER_ADD")


def _lbt_get_moe_scatter_add_bf16():
    global _LBT_MOE_SCATTER_ADD_BF16, _LBT_MOE_SCATTER_ADD_BF16_IMPORT_ATTEMPTED
    if not _LBT_MOE_SCATTER_ADD_BF16_IMPORT_ATTEMPTED:
        _LBT_MOE_SCATTER_ADD_BF16_IMPORT_ATTEMPTED = True
        try:
            from low_bits_training.quantization.mxfp4_backend import (
                mxfp4_moe_scatter_add_bf16,
            )
        except (AttributeError, FileNotFoundError, ImportError):
            _LBT_MOE_SCATTER_ADD_BF16 = None
        else:
            _LBT_MOE_SCATTER_ADD_BF16 = mxfp4_moe_scatter_add_bf16
    return _LBT_MOE_SCATTER_ADD_BF16


def _lbt_local_reduce_index_add_(
    reduced: torch.Tensor,
    global_token_indices: torch.Tensor,
    routed_output: torch.Tensor,
) -> None:
    if (
        _lbt_ep_local_reduce_tk_scatter_add()
        and reduced.is_cuda
        and routed_output.is_cuda
        and global_token_indices.is_cuda
        and reduced.dtype == torch.bfloat16
        and routed_output.dtype == torch.bfloat16
    ):
        scatter_add_fn = _lbt_get_moe_scatter_add_bf16()
        if scatter_add_fn is not None:
            try:
                scatter_add_fn(
                    routed_output.contiguous(),
                    global_token_indices.to(torch.int64).contiguous(),
                    reduced,
                )
                return
            except (AttributeError, FileNotFoundError, ImportError):
                pass
    reduced.index_add_(0, global_token_indices.to(torch.int64), routed_output)


def _lbt_quantize_fp8_for_a2a(
    tensor: torch.Tensor,
    mode: str,
    group,
) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_dtype = _lbt_fp8_dtype(mode)
    fp8_max = torch.finfo(fp8_dtype).max
    if _lbt_ep_a2a_scale_mode() == "static":
        scale = _lbt_static_fp8_scale(tensor.device)
    elif tensor.numel() == 0:
        scale = torch.tensor(1.0, device=tensor.device, dtype=torch.float32)
    else:
        amax = tensor.detach().abs().amax().float()
        scale = torch.clamp(amax / fp8_max, min=1.0e-12)

    scales = _lbt_gather_ep_scales(scale, group)
    payload = torch.clamp(tensor / scale, min=-fp8_max, max=fp8_max).to(fp8_dtype)
    return payload.contiguous(), scales


def _lbt_apply_source_scales(
    tensor: torch.Tensor,
    output_splits: list[int] | None,
    scales: torch.Tensor,
) -> torch.Tensor:
    if output_splits is None:
        output_splits = _lbt_equal_splits(int(tensor.shape[0]), int(scales.numel()))
    scales = scales.to(device=tensor.device, dtype=tensor.dtype)
    offset = 0
    for source_rank, split in enumerate(output_splits):
        split = int(split)
        if split > 0:
            tensor.narrow(0, offset, split).mul_(scales[source_rank])
        offset += split
    return tensor


def _lbt_all_to_all_no_autograd(
    tensor: torch.Tensor,
    output_splits: list[int] | None,
    input_splits: list[int] | None,
    group,
) -> torch.Tensor:
    output_shape = (
        tuple(tensor.shape)
        if output_splits is None
        else (sum(output_splits), *tensor.shape[1:])
    )
    output = torch.empty(output_shape, device=tensor.device, dtype=tensor.dtype)
    torch.distributed.all_to_all_single(
        output,
        tensor.contiguous(),
        output_split_sizes=output_splits,
        input_split_sizes=input_splits,
        group=group,
    )
    return output


def _lbt_compressed_all_to_all(
    tensor: torch.Tensor,
    output_splits: list[int] | None,
    input_splits: list[int] | None,
    group,
    mode: str,
) -> torch.Tensor:
    if mode == "none" or not _lbt_should_compress_a2a_tensor(tensor):
        return _lbt_all_to_all_no_autograd(tensor, output_splits, input_splits, group)

    output_dtype = tensor.dtype
    if _lbt_ep_a2a_scale_mode() == "static":
        fp8_dtype = _lbt_fp8_dtype(mode)
        fp8_max = torch.finfo(fp8_dtype).max
        scale = _lbt_static_fp8_scale_value()
        if scale == 1.0:
            payload = torch.clamp(tensor, min=-fp8_max, max=fp8_max).to(fp8_dtype)
        else:
            payload = torch.clamp(tensor / scale, min=-fp8_max, max=fp8_max).to(
                fp8_dtype
            )
        output = _lbt_all_to_all_no_autograd(
            payload.contiguous(),
            output_splits,
            input_splits,
            group,
        ).to(output_dtype)
        if scale != 1.0:
            output.mul_(scale)
        return output

    payload, source_scales = _lbt_quantize_fp8_for_a2a(tensor, mode, group)
    output = _lbt_all_to_all_no_autograd(
        payload,
        output_splits,
        input_splits,
        group,
    ).to(output_dtype)
    return _lbt_apply_source_scales(output, output_splits, source_scales)


class _LBTCompressedAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        output_splits: list[int] | None,
        input_splits: list[int] | None,
        group,
        forward_mode: str,
        backward_mode: str,
    ) -> torch.Tensor:
        ctx.output_splits = (
            None if output_splits is None else [int(split) for split in output_splits]
        )
        ctx.input_splits = (
            None if input_splits is None else [int(split) for split in input_splits]
        )
        ctx.group = group
        ctx.backward_mode = backward_mode
        return _lbt_compressed_all_to_all(
            tensor,
            ctx.output_splits,
            ctx.input_splits,
            group,
            forward_mode,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_input = _lbt_compressed_all_to_all(
            grad_output,
            ctx.input_splits,
            ctx.output_splits,
            ctx.group,
            ctx.backward_mode,
        )
        return grad_input, None, None, None, None, None


# Diagnostic local-reduce variant: keep the reduce-scatter forward, but use a
# custom backward to compare token-granular A2A against the default all-gather.
# A routes mode is retained for diagnostics, but it sends top_k-expanded rows.
class _LBTLocalReduceCombine(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        routed_output: torch.Tensor,
        local_token_indices: torch.Tensor,
        source_token_indices: torch.Tensor,
        input_splits: list[int],
        output_splits: list[int],
        num_origin_tokens: int,
        group,
    ) -> torch.Tensor:
        input_splits = [int(split) for split in input_splits]
        output_splits = [int(split) for split in output_splits]
        num_origin_tokens = int(num_origin_tokens)
        ep_degree = len(input_splits)

        global_token_indices = torch.empty_like(local_token_indices)
        offset = 0
        for source_rank, split in enumerate(output_splits):
            split = int(split)
            if split > 0:
                global_token_indices[offset : offset + split] = (
                    local_token_indices[offset : offset + split]
                    + source_rank * num_origin_tokens
                )
            offset += split

        reduced = routed_output.new_zeros(
            (ep_degree * num_origin_tokens, routed_output.shape[1])
        )
        reduced.index_add_(0, global_token_indices.to(torch.int64), routed_output)
        output = reduce_scatter_tensor_autograd(
            reduced.contiguous(),
            "sum",
            0,
            group,
        )

        ctx.save_for_backward(local_token_indices, source_token_indices)
        ctx.input_splits = input_splits
        ctx.output_splits = output_splits
        ctx.num_origin_tokens = num_origin_tokens
        ctx.backward_mode = _lbt_ep_local_reduce_a2a_bwd_mode()
        ctx.group = group
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        local_token_indices, source_token_indices = ctx.saved_tensors
        if ctx.backward_mode == "routes":
            send_grad = grad_output.contiguous().index_select(
                0,
                source_token_indices.to(torch.int64),
            )
            grad_routed_output = _lbt_all_to_all_no_autograd(
                send_grad,
                ctx.output_splits,
                ctx.input_splits,
                ctx.group,
            )
        else:
            ep_degree = len(ctx.input_splits)
            token_splits = [int(ctx.num_origin_tokens) for _ in range(ep_degree)]
            send_grad = grad_output.contiguous().repeat(ep_degree, 1)
            gathered_grad = _lbt_all_to_all_no_autograd(
                send_grad,
                token_splits,
                token_splits,
                ctx.group,
            )
            global_token_indices = torch.empty_like(local_token_indices)
            offset = 0
            for source_rank, split in enumerate(ctx.output_splits):
                split = int(split)
                if split > 0:
                    global_token_indices[offset : offset + split] = (
                        local_token_indices[offset : offset + split]
                        + source_rank * int(ctx.num_origin_tokens)
                    )
                offset += split
            grad_routed_output = gathered_grad.index_select(
                0,
                global_token_indices.to(torch.int64),
            )
        return grad_routed_output, None, None, None, None, None, None


def _lbt_all_to_all_single_autograd(
    tensor: torch.Tensor,
    output_splits: list[int] | None,
    input_splits: list[int] | None,
    group,
) -> torch.Tensor:
    forward_mode = _lbt_ep_a2a_mode()
    if forward_mode == "none":
        return all_to_all_single_autograd(
            tensor,
            output_splits,
            input_splits,
            group,
        )
    if not _lbt_should_compress_a2a_tensor(tensor):
        if _lbt_ep_a2a_custom_autograd_below_threshold():
            return _LBTCompressedAllToAll.apply(
                tensor,
                output_splits,
                input_splits,
                group,
                "none",
                "none",
            )
        return all_to_all_single_autograd(
            tensor,
            output_splits,
            input_splits,
            group,
        )

    backward_mode = _lbt_ep_a2a_mode("LBT_EP_A2A_COMPRESS_BWD")
    return _LBTCompressedAllToAll.apply(
        tensor,
        output_splits,
        input_splits,
        group,
        forward_mode,
        backward_mode,
    )


def _lbt_debug_ep_splits(
    phase: str,
    tensor: torch.Tensor,
    input_splits: list[int],
    output_splits: list[int],
    device_mesh: DeviceMesh,
):
    global _LBT_EP_DEBUG_SPLIT_COUNT
    if not _lbt_env_flag("LBT_EP_DEBUG_SPLITS"):
        return
    if _LBT_EP_DEBUG_SPLIT_COUNT >= _lbt_ep_debug_split_limit():
        return

    call_idx = _LBT_EP_DEBUG_SPLIT_COUNT
    _LBT_EP_DEBUG_SPLIT_COUNT += 1

    try:
        rank = torch.distributed.get_rank()
    except Exception:
        rank = -1
    try:
        ep_rank = device_mesh.get_local_rank()
    except Exception:
        ep_rank = -1

    in_total, in_max, in_ratio = _lbt_split_stats(input_splits)
    out_total, out_max, out_ratio = _lbt_split_stats(output_splits)
    row_elems = tensor.numel() // max(int(tensor.shape[0]), 1)
    elem_size = tensor.element_size()
    in_mib = in_total * row_elems * elem_size / (1024 * 1024)
    out_mib = out_total * row_elems * elem_size / (1024 * 1024)

    print(
        "[lbt_ep_debug] "
        f"call={call_idx} phase={phase} rank={rank} ep_rank={ep_rank} "
        f"a2a_compress={_lbt_ep_a2a_mode()} "
        f"dtype={tensor.dtype} shape={tuple(tensor.shape)} row_elems={row_elems} "
        f"in_total={in_total} in_max={in_max} in_skew={in_ratio:.3f} in_mib={in_mib:.2f} "
        f"out_total={out_total} out_max={out_max} out_skew={out_ratio:.3f} out_mib={out_mib:.2f} "
        f"input_splits={input_splits} output_splits={output_splits}",
        flush=True,
    )


# implementation of Tensor Parallel for the GroupedExperts in MoE
class TensorParallel(ParallelStyle):
    def _prepare_input_fn(self, mod, inputs, device_mesh):
        routed_input, num_tokens_per_expert = inputs
        # NOTE: Currently in MoE TP, experts multiplication runs in plain Tensors.
        #       The grad_placements on inputs is set to Partial so that necessary
        #       reductions are performed during backward.
        routed_input = DTensor.from_local(
            routed_input, device_mesh, (Replicate(),)
        ).to_local(grad_placements=(Partial(),))

        return routed_input, num_tokens_per_expert

    def _partition_fn(self, name, module, device_mesh):
        # w1 shape = (experts, out_dim, in_dim)
        module.register_parameter(
            "w1", nn.Parameter(distribute_tensor(module.w1, device_mesh, [Shard(1)]))
        )  # Column-wise sharding

        # w2 shape = (experts, in_dim, out_dim)
        module.register_parameter(
            "w2",
            nn.Parameter(distribute_tensor(module.w2, device_mesh, [Shard(2)])),
        )  # Row-wise sharding

        # w3 shape = (experts, out_dim, in_dim)
        module.register_parameter(
            "w3",
            nn.Parameter(distribute_tensor(module.w3, device_mesh, [Shard(1)])),
        )  # Column-wise sharding

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            self._partition_fn,
            self._prepare_input_fn,
        )


class ExpertParallel(ParallelStyle):
    def __init__(self):
        super().__init__()
        self.input_splits = None
        self.output_splits = None
        self._equal_split_a2a = False
        self.input_shape = None
        self.permuted_indices = None

    # performing all-to-all dispatch on the input
    def _token_dispatch(self, mod, inputs, device_mesh):
        # annotate module input placements/sharding with input_layouts
        routed_input, num_tokens_per_expert = inputs
        if _lbt_env_flag("LBT_MOE_ROUTE_METADATA_DEBUG"):
            with torch.no_grad():
                counts_debug = num_tokens_per_expert
                to_local = getattr(counts_debug, "to_local", None)
                if to_local is not None:
                    try:
                        counts_debug = to_local()
                    except Exception:
                        pass
                counts_debug_i64 = counts_debug.to(torch.int64)
                print(
                    "[lbt_ep_counts]"
                    f" rank={torch.distributed.get_rank()}"
                    f" type={type(num_tokens_per_expert).__name__}"
                    f" is_dtensor={int(isinstance(num_tokens_per_expert, DTensor))}"
                    f" shape={tuple(num_tokens_per_expert.shape)}"
                    f" local_shape={tuple(counts_debug_i64.shape)}"
                    f" sum={int(counts_debug_i64.sum().item())}"
                    f" counts={counts_debug_i64.tolist()}",
                    flush=True,
                )
        if num_tokens_per_expert.dtype not in (torch.int32, torch.int64):
            num_tokens_per_expert = num_tokens_per_expert.to(torch.int64)
        ep_degree = device_mesh.shape[0]
        num_local_experts = num_tokens_per_expert.shape[0] // ep_degree
        total_routed_rows = int(routed_input.shape[0])
        equal_split_a2a = (
            ep_degree == 2
            and total_routed_rows % ep_degree == 0
            and _lbt_ep_exact_equal_split_a2a()
        )

        # generate the input splits and output splits for all-to-all
        with torch.no_grad():
            if not equal_split_a2a:
                input_split_sizes = num_tokens_per_expert.view(ep_degree, -1).sum(dim=1)
            if _lbt_ep_split_count_all_gather():
                gathered_counts = torch.empty(
                    ep_degree * num_tokens_per_expert.numel(),
                    device=num_tokens_per_expert.device,
                    dtype=num_tokens_per_expert.dtype,
                )
                torch.distributed.all_gather_into_tensor(
                    gathered_counts,
                    num_tokens_per_expert.contiguous(),
                    group=device_mesh.get_group(),
                )
                gathered_counts = gathered_counts.view(ep_degree, -1)
                local_expert_start = device_mesh.get_local_rank() * num_local_experts
                num_tokens_per_expert_group = gathered_counts[
                    :,
                    local_expert_start : local_expert_start + num_local_experts,
                ].reshape(-1)
            else:
                num_tokens_per_expert_group = all_to_all_single(
                    num_tokens_per_expert,
                    None,
                    None,
                    group=device_mesh.get_group(),
                )
                # Need to wait explicitly because it is used by a triton kernel later
                # which doesn't realize that AsyncCollectiveTensor needs unwrapping
                num_tokens_per_expert_group = torch.ops._c10d_functional.wait_tensor(
                    num_tokens_per_expert_group
                )
            alignment = int(moe_utils.TOKEN_GROUP_ALIGN_SIZE_M)
            local_expert_counts = num_tokens_per_expert_group.view(ep_degree, -1).sum(dim=0)
            local_expert_counts = torch.clamp_min(local_expert_counts, alignment)
            local_expert_counts = (
                (local_expert_counts + alignment - 1) // alignment * alignment
            )
            local_expert_counts_i32 = local_expert_counts.to(torch.int32)
            if equal_split_a2a:
                self.input_splits = _lbt_equal_splits(total_routed_rows, ep_degree)
                self.output_splits = list(self.input_splits)
                # Only local expert counts need a host list in exact equal-split
                # mode; the activation all-to-all can use implicit equal splits.
                local_expert_counts_list = (
                    local_expert_counts.to(torch.int64)
                    .to(torch.device("cpu"), non_blocking=False)
                    .tolist()
                )
            else:
                split_sizes = torch.stack(
                    (
                        input_split_sizes,
                        num_tokens_per_expert_group.view(ep_degree, -1).sum(dim=1),
                    )
                )
                # all_to_all_single requires host split lists. Copy dispatch splits
                # and the post-permute local expert counts together so downstream
                # grouped experts do not pay another tiny D2H synchronization.
                host_sizes = torch.cat(
                    (
                        split_sizes.reshape(-1).to(torch.int64),
                        local_expert_counts.reshape(-1).to(torch.int64),
                    )
                ).to(torch.device("cpu"), non_blocking=False)
                self.input_splits = host_sizes[:ep_degree].tolist()
                self.output_splits = host_sizes[ep_degree : 2 * ep_degree].tolist()
                local_expert_counts_list = host_sizes[2 * ep_degree :].tolist()
            self._equal_split_a2a = equal_split_a2a

        _lbt_debug_ep_splits(
            "dispatch",
            routed_input,
            self.input_splits,
            self.output_splits,
            device_mesh,
        )

        # perform all-to-all
        routed_input = _lbt_all_to_all_single_autograd(
            routed_input,
            None if self._equal_split_a2a else self.output_splits,
            None if self._equal_split_a2a else self.input_splits,
            device_mesh.get_group(),
        )

        # NOTE: After this all-to-all, the routed input is put on proper EP rank.
        # However, the num_tokens_per_expert_group is not of the final target format
        # [#tokens for local expert 0, #tokens for local expert 1, ...]
        # Rather, it is of the format
        # [#tokens for local expert 0 from EP rank 0, #tokens for local expert 1 from EP rank 0, ...,
        #  #tokens for local expert 0 from EP rank 1, #tokens for local expert 1 from EP rank 1, ...]
        # We need to perform another shuffle to get the correct layout, via the _permute function
        # below, which also does padding to make sure the number of tokens each expert gets locally
        # is a multiple of TOKEN_GROUP_ALIGN_SIZE_M.
        # Note that this will create side effects when wrapping the for-loop implementation
        # of GroupedExperts, as it does not need padding.

        (
            self.input_shape,
            routed_input,
            self.permuted_indices,
            num_tokens_per_expert_group,
        ) = _permute(
            routed_input,
            num_tokens_per_expert_group,
            ep_degree,
            num_local_experts,
            local_expert_counts_i32,
        )
        try:
            num_tokens_per_expert_group._lbt_counts_list = local_expert_counts_list
        except Exception:
            pass

        return routed_input, num_tokens_per_expert_group

    @staticmethod
    def _partition_fn(name, mod, device_mesh):
        # shard on the expert dimension
        for name, param in mod.named_parameters(recurse=False):
            dist_param = nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)]))
            mod.register_parameter(name, dist_param)

    # performing all-to-all combine on the output
    def _token_combine(self, mod, routed_output, device_mesh):
        routed_output = _unpermute(
            routed_output, self.input_shape, self.permuted_indices
        )

        _lbt_debug_ep_splits(
            "combine",
            routed_output,
            self.output_splits,
            self.input_splits,
            device_mesh,
        )

        routed_output = _lbt_all_to_all_single_autograd(
            routed_output,
            None if self._equal_split_a2a else self.input_splits,
            None if self._equal_split_a2a else self.output_splits,
            device_mesh.get_group(),
        )
        return routed_output

    def _dispatch_like_current_routes(
        self,
        tensor: torch.Tensor,
        device_mesh: DeviceMesh,
    ) -> torch.Tensor:
        if tensor.dim() == 1:
            tensor = tensor.reshape(-1, 1)
        routed = _lbt_all_to_all_single_autograd(
            tensor,
            None if self._equal_split_a2a else self.output_splits,
            None if self._equal_split_a2a else self.input_splits,
            device_mesh.get_group(),
        )
        routed = torch.vstack((routed, routed.new_zeros((routed.shape[-1],))))
        return routed[self.permuted_indices, :]

    def _dispatch_metadata_like_current_routes(
        self,
        tensor: torch.Tensor,
        device_mesh: DeviceMesh,
    ) -> torch.Tensor:
        if tensor.dim() == 1:
            tensor = tensor.reshape(-1, 1)
        routed = _lbt_all_to_all_no_autograd(
            tensor.contiguous(),
            None if self._equal_split_a2a else self.output_splits,
            None if self._equal_split_a2a else self.input_splits,
            device_mesh.get_group(),
        )
        routed = torch.vstack((routed, routed.new_zeros((routed.shape[-1],))))
        return routed[self.permuted_indices, :]

    def _token_combine_local_reduced(
        self,
        routed_output: torch.Tensor,
        local_token_indices: torch.Tensor,
        source_token_indices: torch.Tensor | None,
        num_origin_tokens: int,
        device_mesh: DeviceMesh,
    ) -> torch.Tensor:
        ep_degree = int(device_mesh.shape[0])
        num_origin_tokens = int(num_origin_tokens)
        if (
            _lbt_ep_local_reduce_permuted()
            and not _lbt_ep_local_reduce_a2a_bwd()
            and int(local_token_indices.numel()) == int(routed_output.shape[0])
            and int(self.permuted_indices.numel()) == int(routed_output.shape[0])
        ):
            num_routed_rows = sum(self.output_splits)
            source_offsets = torch.empty(
                num_routed_rows + 1,
                device=local_token_indices.device,
                dtype=local_token_indices.dtype,
            )
            offset = 0
            for source_rank, split in enumerate(self.output_splits):
                split = int(split)
                if split > 0:
                    source_offsets[offset : offset + split] = (
                        source_rank * num_origin_tokens
                    )
                offset += split
            source_offsets[num_routed_rows] = 0
            source_offsets = source_offsets[self.permuted_indices]
            global_token_indices = local_token_indices.reshape(-1) + source_offsets

            reduced = routed_output.new_zeros(
                (ep_degree * num_origin_tokens, routed_output.shape[1])
            )
            _lbt_local_reduce_index_add_(
                reduced,
                global_token_indices,
                routed_output,
            )

            token_splits = [num_origin_tokens for _ in range(ep_degree)]
            _lbt_debug_ep_splits(
                "combine_reduced_permuted",
                reduced,
                token_splits,
                token_splits,
                device_mesh,
            )
            if _lbt_ep_local_reduce_collective() == "reduce_scatter":
                return reduce_scatter_tensor_autograd(
                    reduced.contiguous(),
                    "sum",
                    0,
                    device_mesh.get_group(),
                )
            partials = _lbt_all_to_all_single_autograd(
                reduced,
                token_splits,
                token_splits,
                device_mesh.get_group(),
            )
            return partials.view(ep_degree, num_origin_tokens, -1).sum(dim=0)

        row_input_shape = (self.input_shape[0], 1)
        routed_output = _unpermute(
            routed_output,
            (self.input_shape[0], routed_output.shape[1]),
            self.permuted_indices,
        )
        local_token_indices = _unpermute(
            local_token_indices.reshape(-1, 1),
            row_input_shape,
            self.permuted_indices,
        ).reshape(-1)

        if (
            _lbt_ep_local_reduce_collective() == "reduce_scatter"
            and _lbt_ep_local_reduce_a2a_bwd()
            and source_token_indices is not None
            and int(source_token_indices.numel()) == sum(self.input_splits)
        ):
            return _LBTLocalReduceCombine.apply(
                routed_output,
                local_token_indices,
                source_token_indices.reshape(-1).to(torch.int64).contiguous(),
                self.input_splits,
                self.output_splits,
                num_origin_tokens,
                device_mesh.get_group(),
            )

        global_token_indices = torch.empty_like(local_token_indices)
        offset = 0
        for source_rank, split in enumerate(self.output_splits):
            split = int(split)
            if split > 0:
                global_token_indices[offset : offset + split] = (
                    local_token_indices[offset : offset + split]
                    + source_rank * num_origin_tokens
                )
            offset += split

        reduced = routed_output.new_zeros(
            (ep_degree * num_origin_tokens, routed_output.shape[1])
        )
        _lbt_local_reduce_index_add_(
            reduced,
            global_token_indices,
            routed_output,
        )

        token_splits = [num_origin_tokens for _ in range(ep_degree)]
        _lbt_debug_ep_splits(
            "combine_reduced",
            reduced,
            token_splits,
            token_splits,
            device_mesh,
        )
        if _lbt_ep_local_reduce_collective() == "reduce_scatter":
            return reduce_scatter_tensor_autograd(
                reduced.contiguous(),
                "sum",
                0,
                device_mesh.get_group(),
            )
        partials = _lbt_all_to_all_single_autograd(
            reduced,
            token_splits,
            token_splits,
            device_mesh.get_group(),
        )
        return partials.view(ep_degree, num_origin_tokens, -1).sum(dim=0)

    def _forward_ep_scored_output_combine(
        self,
        mod: nn.Module,
        routed_input: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
        top_scores: torch.Tensor,
        token_indices: torch.Tensor | None,
        num_origin_tokens: int | None,
        device_mesh: DeviceMesh,
    ) -> torch.Tensor | None:
        if not _lbt_ep_scored_output_combine():
            return None
        if routed_input.dim() != 2 or top_scores.dim() != 1:
            return None
        if int(routed_input.shape[0]) != int(top_scores.numel()):
            return None

        local_token_indices = None
        pack_local_reduce_indices = (
            _lbt_ep_local_reduce_output_combine()
            and _lbt_ep_pack_local_reduce_indices()
            and token_indices is not None
            and num_origin_tokens is not None
            and int(num_origin_tokens) <= 256 * 256
            and int(token_indices.numel()) == int(top_scores.numel())
        )
        score_dispatch_mode = _lbt_ep_score_dispatch_mode()
        if score_dispatch_mode == "pack_bf16":
            packed_cols = [routed_input, top_scores.reshape(-1, 1).to(routed_input.dtype)]
            if pack_local_reduce_indices:
                token_indices_i64 = token_indices.reshape(-1).to(torch.int64)
                packed_cols.extend(
                    (
                        torch.remainder(token_indices_i64, 256)
                        .reshape(-1, 1)
                        .to(routed_input.dtype),
                        torch.div(token_indices_i64, 256, rounding_mode="floor")
                        .reshape(-1, 1)
                        .to(routed_input.dtype),
                    )
                )
            routed_input_and_scores = torch.cat(packed_cols, dim=1)
            local_input_and_scores, local_counts = self._token_dispatch(
                mod,
                (routed_input_and_scores, num_tokens_per_expert),
                device_mesh,
            )
            if pack_local_reduce_indices:
                local_scores = local_input_and_scores[:, -3:-2].to(torch.float32)
                local_token_indices = (
                    local_input_and_scores[:, -2].to(torch.int64)
                    + local_input_and_scores[:, -1].to(torch.int64) * 256
                )
                local_input = local_input_and_scores[:, :-3].contiguous()
            else:
                local_scores = local_input_and_scores[:, -1:].to(torch.float32)
                local_input = local_input_and_scores[:, :-1].contiguous()
        else:
            local_input, local_counts = self._token_dispatch(
                mod,
                (routed_input, num_tokens_per_expert),
                device_mesh,
            )
            score_dtype = (
                routed_input.dtype
                if score_dispatch_mode == "separate_bf16"
                else torch.float32
            )
            local_scores = self._dispatch_like_current_routes(
                top_scores.reshape(-1, 1).to(score_dtype),
                device_mesh,
            ).to(torch.float32)

        local_output = mod.forward(local_input, local_counts)
        local_output = (local_output.to(torch.float32) * local_scores).to(
            local_output.dtype
        )
        if (
            _lbt_ep_local_reduce_output_combine()
            and token_indices is not None
            and num_origin_tokens is not None
        ):
            if local_token_indices is None:
                index_dtype = torch.int64
                if _lbt_ep_local_reduce_index_dtype() == "auto":
                    max_global_index = int(device_mesh.shape[0]) * int(num_origin_tokens)
                    if max_global_index <= torch.iinfo(torch.int32).max:
                        index_dtype = torch.int32
                local_token_indices = self._dispatch_metadata_like_current_routes(
                    token_indices.reshape(-1).to(index_dtype),
                    device_mesh,
                )
            return self._token_combine_local_reduced(
                local_output,
                local_token_indices,
                token_indices.reshape(-1),
                int(num_origin_tokens),
                device_mesh,
            )
        if score_dispatch_mode == "pack_bf16":
            self.input_shape = (self.input_shape[0], local_output.shape[1])
        return self._token_combine(mod, local_output, device_mesh)

    def _attach_scored_output_combine(
        self,
        module: nn.Module,
        device_mesh: DeviceMesh,
    ) -> None:
        ep_style = self

        def forward_ep_scored_output_combine(
            mod: nn.Module,
            routed_input: torch.Tensor,
            num_tokens_per_expert: torch.Tensor,
            top_scores: torch.Tensor,
            token_indices: torch.Tensor | None = None,
            num_origin_tokens: int | None = None,
        ) -> torch.Tensor | None:
            return ep_style._forward_ep_scored_output_combine(
                mod,
                routed_input,
                num_tokens_per_expert,
                top_scores,
                token_indices,
                num_origin_tokens,
                device_mesh,
            )

        module.forward_ep_scored_output_combine = types.MethodType(
            forward_ep_scored_output_combine,
            module,
        )

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        module = distribute_module(
            module,
            device_mesh,
            partition_fn=ExpertParallel._partition_fn,
            input_fn=self._token_dispatch,
            output_fn=self._token_combine,
        )
        self._attach_scored_output_combine(module, device_mesh)
        return module


# This class is for dp2ep with TP (without TP we can just use ExpertParallel)
class ExpertTensorParallel(ExpertParallel):
    def _token_dispatch(self, mod, inputs, device_mesh):
        routed_input, num_tokens_per_expert = inputs

        # NOTE: Currently in MoE TP, experts multiplication runs in plain Tensors.
        #       The grad_placements on inputs is set to Partial so that necessary
        #       reductions are performed during backward.
        routed_input = DTensor.from_local(
            routed_input, device_mesh["tp"], (Replicate(),)
        ).to_local(grad_placements=(Partial(),))

        inputs = (routed_input, num_tokens_per_expert)

        # token dispatch happens on the EP mesh, whereas device_mesh is [ep, tp] mesh
        return super()._token_dispatch(mod, inputs, device_mesh["ep"])

    def _partition_fn_2d(self, name, mod, ep_tp_mesh):
        # w1 shape = (experts, out_dim, in_dim)
        mod.register_parameter(
            "w1",
            nn.Parameter(distribute_tensor(mod.w1, ep_tp_mesh, [Shard(0), Shard(1)])),
        )  # Column-wise sharding

        # w2 shape = (experts, in_dim, out_dim)
        mod.register_parameter(
            "w2",
            nn.Parameter(distribute_tensor(mod.w2, ep_tp_mesh, [Shard(0), Shard(2)])),
        )  # Row-wise sharding

        # w3 shape = (experts, out_dim, in_dim)
        mod.register_parameter(
            "w3",
            nn.Parameter(distribute_tensor(mod.w3, ep_tp_mesh, [Shard(0), Shard(1)])),
        )  # Column-wise sharding

    def _token_combine(self, mod, routed_output, device_mesh):
        # token combine happens on the EP mesh, whereas device_mesh is [ep, tp] mesh
        return super()._token_combine(mod, routed_output, device_mesh["ep"])

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            partition_fn=self._partition_fn_2d,
            input_fn=self._token_dispatch,
            output_fn=self._token_combine,
        )


# This class is to support Sequence Parallel for ETP=1
# when EP borrows from all TP and part of DP
class ReordererSequenceParallel(ParallelStyle):
    def __init__(self):
        super().__init__()

    def _prepare_inputput_fn(self, mod, inputs, device_mesh):
        # shape (batch_size*seq_len, top_k)
        top_scores, selected_experts_indices = inputs
        num_tokens, _ = top_scores.shape

        # NOTE: If needed, we can pad tokens in case bs*slen is not divisible by TP degree
        # if top_scores.shape[0] % device_mesh.size() != 0:
        #     num_tokens = top_scores.shape[0]
        #     tp_size = device_mesh.size()
        #     n_pad = (num_tokens // tp_size + 1) * tp_size - num_tokens
        #     selected_experts_indices = F.pad(selected_experts_indices, [0, 0, 0, n_pad])
        #     top_scores = F.pad(top_scores, [0, 0, 0, n_pad])

        def _split_along_first_dim(x: torch.Tensor) -> torch.Tensor:
            assert x.is_contiguous()
            if num_tokens % device_mesh.size() != 0:
                raise ValueError(
                    "Uneven split of tokens of is not supported yet. "
                    "Requires EP degree dividing batch size * seq len."
                )
            local_num_tokens = num_tokens // device_mesh.size()
            local_rank = device_mesh.get_local_rank()
            offset = local_rank * local_num_tokens
            output = x[offset : offset + local_num_tokens]

            return output

        top_scores = _split_along_first_dim(top_scores)
        selected_experts_indices = _split_along_first_dim(selected_experts_indices)

        # shape (batch_size * seq_len // ep_degree, top_k)
        return top_scores, selected_experts_indices

    def _prepare_output_fn(self, mod, outputs, device_mesh):
        # shape (batch_size * seq_len * top_k // ep_degree)
        top_scores, token_indices_experts_sorted, num_tokens_per_expert = outputs

        # NOTE: As we shard routed tokens along bs*slen dim across the TP ranks,
        #       the MoE gather and scatter still require global token indices.
        local_rank = device_mesh.get_local_rank()
        # fact: top_scores.shape[0] // mod.top_k = batch_size * seq_len // ep_degree
        if not hasattr(mod, "top_k"):
            raise ValueError(
                "TokenReorderer class in MoE should always have top_k attribute."
            )
        token_indices_experts_sorted += top_scores.shape[0] // mod.top_k * local_rank

        return top_scores, token_indices_experts_sorted, num_tokens_per_expert

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            partition_fn=None,
            input_fn=self._prepare_inputput_fn,
            output_fn=self._prepare_output_fn,
        )
