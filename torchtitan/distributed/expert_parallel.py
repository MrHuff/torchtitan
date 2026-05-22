# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch
import torch.nn as nn
from torch.distributed._functional_collectives import (
    all_to_all_single,
    all_to_all_single_autograd,
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
    return _lbt_env_flag("LBT_EP_SPLIT_COUNT_ALL_GATHER", False)


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
    output_splits: list[int],
    scales: torch.Tensor,
) -> torch.Tensor:
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
    output_splits: list[int],
    input_splits: list[int],
    group,
) -> torch.Tensor:
    output_shape = (sum(output_splits), *tensor.shape[1:])
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
    output_splits: list[int],
    input_splits: list[int],
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
        output_splits: list[int],
        input_splits: list[int],
        group,
        forward_mode: str,
        backward_mode: str,
    ) -> torch.Tensor:
        ctx.output_splits = [int(split) for split in output_splits]
        ctx.input_splits = [int(split) for split in input_splits]
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


def _lbt_all_to_all_single_autograd(
    tensor: torch.Tensor,
    output_splits: list[int],
    input_splits: list[int],
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

        # generate the input splits and output splits for all-to-all
        with torch.no_grad():
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
            split_sizes = torch.stack(
                (
                    input_split_sizes,
                    num_tokens_per_expert_group.view(ep_degree, -1).sum(dim=1),
                )
            )
            alignment = int(moe_utils.TOKEN_GROUP_ALIGN_SIZE_M)
            local_expert_counts = num_tokens_per_expert_group.view(ep_degree, -1).sum(dim=0)
            local_expert_counts = torch.clamp_min(local_expert_counts, alignment)
            local_expert_counts = (
                (local_expert_counts + alignment - 1) // alignment * alignment
            )
            local_expert_counts_i32 = local_expert_counts.to(torch.int32)
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
            self.output_splits,
            self.input_splits,
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
            self.input_splits,
            self.output_splits,
            device_mesh.get_group(),
        )
        return routed_output

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            partition_fn=ExpertParallel._partition_fn,
            input_fn=self._token_dispatch,
            output_fn=self._token_combine,
        )


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
