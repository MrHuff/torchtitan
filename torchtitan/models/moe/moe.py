# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor

from .utils import indices_padding_wrapper

_MXFP4_MOE_REORDER_FN = None
_MXFP4_MOE_REORDER_SCORES_FULL_FN = None
_MXFP4_MOE_EP2_BALANCE_TOP3_ROUTES_FN = None
_MXFP4_MOE_EP2_SELECT_TOP3_BALANCE_SCORES_FN = None
_MXFP4_MOE_EP2_SCATTER_TOP3_SCORES_FN = None
_MXFP4_MOE_GATHER_SCORES_FN = None
_MXFP4_MOE_SCATTER_SCORES_FN = None
_MXFP4_MOE_BUILD_ROUTE_INVERSE_FN = None
_MXFP4_MOE_ROUTE_COMBINE_ADD_FN = None
_MXFP4_MOE_SCALE_SCATTER_ADD_FN = None
_MXFP4_MOE_INDEXED_DOT_ROWS_FN = None
_MXFP4_MOE_INDEXED_SCALE_DOT_ROWS_FN = None
_MXFP4_MOE_INDEXED_SCALE_ROWS_FN = None
_MXFP4_MOE_REORDER_IMPORT_ATTEMPTED = False
_LBT_MOE_ROUTER_DEBUG_COUNT = 0
_LBT_MOE_FORWARD_DEBUG_COUNT = 0
_LBT_MOE_ROUTER_INIT_FALLBACK_COUNT = 0


def _lbt_env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def _lbt_env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _lbt_dist_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def _lbt_dist_world_size() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    try:
        return int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError:
        return 1


def _lbt_local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if isinstance(tensor, DTensor):
        return tensor.to_local()
    return tensor


def _lbt_trunc_normal_with_dtensor_fallback_(
    tensor: torch.Tensor,
    *,
    mean: float,
    std: float,
) -> None:
    global _LBT_MOE_ROUTER_INIT_FALLBACK_COUNT

    nn.init.trunc_normal_(tensor, mean=mean, std=std)
    if _lbt_env_flag("LBT_MOE_ROUTER_INIT_TRACE"):
        with torch.no_grad():
            local = _lbt_local_tensor(tensor.detach()).float()
            print(
                "[lbt_router_init_trace]"
                f" rank={_lbt_dist_rank()}"
                f" type={type(tensor).__name__}"
                f" is_dtensor={int(isinstance(tensor, DTensor))}"
                f" dtype={tensor.dtype}"
                f" shape={tuple(tensor.shape)}"
                f" local_shape={tuple(local.shape)}"
                f" std={float(local.std(unbiased=False).item()):.6e}"
                f" absmax={float(local.abs().max().item()):.6e}",
                flush=True,
            )
    if not _lbt_env_flag("LBT_MOE_ROUTER_INIT_FALLBACK", True):
        return

    with torch.no_grad():
        local = _lbt_local_tensor(tensor)
        if local.numel() == 0:
            return
        local_float = local.float()
        observed_std = local_float.std(unbiased=False)
        observed_absmax = local_float.abs().max()
        bad_scale = (
            not bool(torch.isfinite(observed_std).item())
            or not bool(torch.isfinite(observed_absmax).item())
            or float(observed_std.item()) > max(float(std) * 4.0, 1.0e-3)
            or float(observed_absmax.item()) > max(float(std) * 32.0, 0.25)
        )
        if not bad_scale:
            return

        # Direct BF16 trunc_normal_ can leave rare +/-2.0 outliers on this stack.
        # Reinitialize through fp32 scratch when the observed router scale is impossible.
        fresh_float = torch.empty(
            tuple(local.shape),
            dtype=torch.float32,
            device=local.device,
        )
        fallback_idx = _LBT_MOE_ROUTER_INIT_FALLBACK_COUNT
        _LBT_MOE_ROUTER_INIT_FALLBACK_COUNT += 1
        seed = (
            int(torch.initial_seed())
            + 1_000_003 * (fallback_idx + 1)
            + 9_176 * (_lbt_dist_rank() + 1)
        ) % ((1 << 63) - 1)
        generator = torch.Generator(device=local.device)
        generator.manual_seed(seed)
        nn.init.trunc_normal_(fresh_float, mean=mean, std=std, generator=generator)
        fresh_local = fresh_float.to(dtype=local.dtype)
        if isinstance(tensor, DTensor):
            replacement = DTensor.from_local(
                fresh_local,
                device_mesh=tensor.device_mesh,
                placements=tensor.placements,
                run_check=False,
                shape=tensor.shape,
                stride=tensor.stride(),
            )
            tensor.copy_(replacement)
        else:
            tensor.copy_(fresh_local)
        if _lbt_env_flag("LBT_MOE_ROUTER_INIT_TRACE"):
            local_float = _lbt_local_tensor(tensor.detach()).float()
            print(
                "[lbt_router_init_trace]"
                f" rank={_lbt_dist_rank()}"
                " fallback=1"
                f" fresh_std={float(fresh_local.float().std(unbiased=False).item()):.6e}"
                f" fresh_absmax={float(fresh_local.float().abs().max().item()):.6e}"
                f" std={float(local_float.std(unbiased=False).item()):.6e}"
                f" absmax={float(local_float.abs().max().item()):.6e}",
                flush=True,
            )


def _lbt_moe_router_debug(
    router: nn.Module,
    x: torch.Tensor,
    scores: torch.Tensor,
    top_scores: torch.Tensor,
    selected_experts_indices: torch.Tensor,
    expert_bias: torch.Tensor | None = None,
) -> None:
    global _LBT_MOE_ROUTER_DEBUG_COUNT
    if not _lbt_env_flag("LBT_MOE_ROUTER_DEBUG"):
        return
    if _lbt_env_flag("LBT_MOE_ROUTER_DEBUG_RANK0_ONLY", True) and _lbt_dist_rank() != 0:
        return
    name = str(getattr(router, "_lbt_debug_name", ""))
    filters = [
        item.strip()
        for item in os.environ.get("LBT_MOE_ROUTER_DEBUG_FILTER", "").split(",")
        if item.strip()
    ]
    if filters and not any(item in name for item in filters):
        return
    limit = _lbt_env_int("LBT_MOE_ROUTER_DEBUG_LIMIT", 16)
    if limit >= 0 and _LBT_MOE_ROUTER_DEBUG_COUNT >= limit:
        return

    _LBT_MOE_ROUTER_DEBUG_COUNT += 1

    with torch.no_grad():
        counts = torch.histc(
            selected_experts_indices.reshape(-1).float(),
            bins=router.num_experts,
            min=0,
            max=router.num_experts,
        )
        counts_total = counts.sum()
        if float(counts_total.item()) > 0:
            expert_l1_uniform = (
                counts / counts_total - (1.0 / float(counts.numel()))
            ).abs().sum()
        else:
            expert_l1_uniform = counts_total
        ep_degree = _lbt_env_int("LBT_MOE_ROUTER_DEBUG_EP_DEGREE", 0)
        ep_summary = ""
        if ep_degree > 1 and counts.numel() % ep_degree == 0:
            ep_totals = counts.view(ep_degree, counts.numel() // ep_degree).sum(dim=1)
            ep_total = ep_totals.sum()
            if float(ep_total.item()) > 0:
                ep_l1_uniform = (
                    ep_totals / ep_total - (1.0 / float(ep_totals.numel()))
                ).abs().sum()
            else:
                ep_l1_uniform = ep_total
            ep_mean = ep_totals.mean()
            ep_skew = ep_totals.max() / ep_mean if float(ep_mean.item()) > 0 else ep_mean
            ep_summary = (
                f" ep_skew={float(ep_skew.item()):.3f}"
                f" ep_l1_uniform={float(ep_l1_uniform.item()):.6f}"
                f" ep_totals=[{','.join(f'{float(v.item()):.0f}' for v in ep_totals)}]"
            )
        top_count_values, top_count_indices = torch.topk(
            counts,
            k=min(6, counts.numel()),
            largest=True,
        )
        bottom_count_values, bottom_count_indices = torch.topk(
            counts,
            k=min(6, counts.numel()),
            largest=False,
        )
        score_spread = scores.max(dim=1).values - scores.min(dim=1).values
        x_float = x.float()
        input_rms = torch.sqrt(x_float.square().mean())
        input_std = x_float.std(unbiased=False)
        gate_weight = getattr(getattr(router, "gate", None), "weight", None)
        if gate_weight is not None:
            gate_weight_float = _lbt_local_tensor(gate_weight.detach()).float()
            gate_weight_absmax = gate_weight_float.abs().max()
            gate_weight_std = gate_weight_float.std(unbiased=False)
        else:
            gate_weight_absmax = scores.new_tensor(float("nan"))
            gate_weight_std = scores.new_tensor(float("nan"))
        route_scores = scores
        if expert_bias is not None and not getattr(router, "_debug_force_load_balance", False):
            route_scores = route_scores + expert_bias.to(
                device=route_scores.device,
                dtype=route_scores.dtype,
            )
        margin_k = min(
            max(2, int(getattr(router, "top_k", 1)) + 1),
            route_scores.shape[1],
        )
        route_top = torch.topk(route_scores.float(), k=margin_k, dim=1).values
        route_top1_margin = route_top[:, 0] - route_top[:, 1]
        if margin_k > int(getattr(router, "top_k", 1)):
            route_boundary_margin = (
                route_top[:, int(getattr(router, "top_k", 1)) - 1]
                - route_top[:, int(getattr(router, "top_k", 1))]
            )
        else:
            route_boundary_margin = route_top1_margin.new_empty(0)
        route_top1_margin_p01 = torch.quantile(route_top1_margin, 0.01)
        if route_boundary_margin.numel() > 0:
            route_boundary_margin_mean = route_boundary_margin.mean()
            route_boundary_margin_p01 = torch.quantile(route_boundary_margin, 0.01)
        else:
            route_boundary_margin_mean = route_top1_margin.new_tensor(float("nan"))
            route_boundary_margin_p01 = route_top1_margin.new_tensor(float("nan"))
        input_row_zero_frac = (x.abs().sum(dim=1) == 0).float().mean()
        print(
            "[lbt_moe_router]"
            f" call={_LBT_MOE_ROUTER_DEBUG_COUNT - 1}"
            f" rank={_lbt_dist_rank()}"
            f" name={name}"
            f" input_absmax={float(x.abs().max().float().item()):.6e}"
            f" input_rms={float(input_rms.item()):.6e}"
            f" input_std={float(input_std.item()):.6e}"
            f" input_row_zero_frac={float(input_row_zero_frac.item()):.6f}"
            f" gate_weight_absmax={float(gate_weight_absmax.item()):.6e}"
            f" gate_weight_std={float(gate_weight_std.item()):.6e}"
            f" score_absmax={float(scores.abs().max().float().item()):.6e}"
            f" score_std={float(scores.float().std(unbiased=False).item()):.6e}"
            f" score_spread_mean={float(score_spread.float().mean().item()):.6e}"
            f" route_top1_margin_mean={float(route_top1_margin.mean().item()):.6e}"
            f" route_top1_margin_p01={float(route_top1_margin_p01.item()):.6e}"
            f" route_boundary_margin_mean={float(route_boundary_margin_mean.item()):.6e}"
            f" route_boundary_margin_p01={float(route_boundary_margin_p01.item()):.6e}"
            f" score_nan={int(torch.isnan(scores).sum().item())}"
            f" score_inf={int(torch.isinf(scores).sum().item())}"
            f" top_score_mean={float(top_scores.float().mean().item()):.6e}"
            f" count_min={float(counts.min().item()):.0f}"
            f" count_max={float(counts.max().item()):.0f}"
            f" expert_l1_uniform={float(expert_l1_uniform.item()):.6f}"
            f"{ep_summary}"
            f" top=[{','.join(f'{int(i.item())}:{float(v.item()):.0f}' for i, v in zip(top_count_indices, top_count_values))}]"
            f" bottom=[{','.join(f'{int(i.item())}:{float(v.item()):.0f}' for i, v in zip(bottom_count_indices, bottom_count_values))}]",
            flush=True,
        )


def _lbt_moe_debug_name_matches(name: str, env_name: str) -> bool:
    filters = [
        item.strip()
        for item in os.environ.get(env_name, "").split(",")
        if item.strip()
    ]
    return not filters or any(item in name for item in filters)


def _lbt_tensor_debug_summary(label: str, tensor: torch.Tensor | None) -> str:
    if tensor is None:
        return f" {label}=None"
    with torch.no_grad():
        data = tensor.detach()
        finite = torch.isfinite(data)
        finite_count = int(finite.sum().item())
        total = data.numel()
        if total == 0:
            return f" {label}_numel=0"
        finite_data = data[finite].float() if finite_count > 0 else data.new_empty(0).float()
        if finite_count > 0:
            absmax = float(finite_data.abs().max().item())
            mean = float(finite_data.mean().item())
            std = float(finite_data.std(unbiased=False).item())
        else:
            absmax = mean = std = float("nan")
        zero_frac = float((data == 0).float().mean().item())
        row_zero = ""
        if data.dim() == 2:
            row_zero_frac = (data.abs().sum(dim=1) == 0).float().mean()
            row_zero = f" {label}_row_zero_frac={float(row_zero_frac.item()):.6f}"
        return (
            f" {label}_absmax={absmax:.6e}"
            f" {label}_mean={mean:.6e}"
            f" {label}_std={std:.6e}"
            f" {label}_zero_frac={zero_frac:.6f}"
            f" {label}_nan={int(torch.isnan(data).sum().item())}"
            f" {label}_inf={int(torch.isinf(data).sum().item())}"
            f"{row_zero}"
        )


def _lbt_moe_forward_debug(
    moe: nn.Module,
    stage: str,
    *,
    x_raw: torch.Tensor | None = None,
    x_normed: torch.Tensor | None = None,
    norm_weight: torch.Tensor | None = None,
    inv_rms: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
) -> None:
    global _LBT_MOE_FORWARD_DEBUG_COUNT
    if not _lbt_env_flag("LBT_MOE_FORWARD_DEBUG"):
        return
    if _lbt_env_flag("LBT_MOE_FORWARD_DEBUG_RANK0_ONLY", True) and _lbt_dist_rank() != 0:
        return
    name = str(getattr(moe, "_lbt_debug_name", ""))
    if not _lbt_moe_debug_name_matches(name, "LBT_MOE_FORWARD_DEBUG_FILTER"):
        return
    limit = _lbt_env_int("LBT_MOE_FORWARD_DEBUG_LIMIT", 32)
    if limit >= 0 and _LBT_MOE_FORWARD_DEBUG_COUNT >= limit:
        return

    call = _LBT_MOE_FORWARD_DEBUG_COUNT
    _LBT_MOE_FORWARD_DEBUG_COUNT += 1
    print(
        "[lbt_moe_forward]"
        f" call={call}"
        f" rank={_lbt_dist_rank()}"
        f" name={name}"
        f" stage={stage}"
        + _lbt_tensor_debug_summary("x_raw", x_raw)
        + _lbt_tensor_debug_summary("x_normed", x_normed)
        + _lbt_tensor_debug_summary("norm_weight", norm_weight)
        + _lbt_tensor_debug_summary("inv_rms", inv_rms)
        + _lbt_tensor_debug_summary("output", output),
        flush=True,
    )


def _lbt_moe_init_debug(moe: nn.Module, init_std: float) -> None:
    if not _lbt_env_flag("LBT_MOE_INIT_DEBUG"):
        return
    if _lbt_env_flag("LBT_MOE_INIT_DEBUG_RANK0_ONLY", True) and _lbt_dist_rank() != 0:
        return
    name = str(getattr(moe, "_lbt_debug_name", ""))
    if not _lbt_moe_debug_name_matches(name, "LBT_MOE_INIT_DEBUG_FILTER"):
        return
    router = getattr(moe, "router", None)
    gate_weight = getattr(getattr(router, "gate", None), "weight", None)
    if gate_weight is None:
        return
    with torch.no_grad():
        data = _lbt_local_tensor(gate_weight.detach()).float()
        print(
            "[lbt_moe_init]"
            f" rank={_lbt_dist_rank()}"
            f" name={name}"
            f" init_std={float(init_std):.6e}"
            f" gate_weight_type={type(gate_weight).__name__}"
            f" gate_weight_is_dtensor={int(isinstance(gate_weight, DTensor))}"
            f" gate_weight_dtype={gate_weight.dtype}"
            f" gate_weight_shape={tuple(gate_weight.shape)}"
            f" gate_weight_local_shape={tuple(data.shape)}"
            f" gate_weight_absmax={float(data.abs().max().item()):.6e}"
            f" gate_weight_std={float(data.std(unbiased=False).item()):.6e}",
            flush=True,
        )


def _lbt_reset_moved_ffn_norm() -> bool:
    return _lbt_env_flag("MXFP4_DEEPSEEK_RESET_MOVED_FFN_NORM", True)


def _get_mxfp4_moe_reorder_fn():
    global _MXFP4_MOE_REORDER_FN
    _ensure_mxfp4_moe_route_imports()
    return _MXFP4_MOE_REORDER_FN


def _get_mxfp4_moe_reorder_scores_full_fn():
    _ensure_mxfp4_moe_route_imports()
    return _MXFP4_MOE_REORDER_SCORES_FULL_FN


def _get_mxfp4_moe_ep2_balance_top3_routes_fn():
    _ensure_mxfp4_moe_route_imports()
    return _MXFP4_MOE_EP2_BALANCE_TOP3_ROUTES_FN


def _get_mxfp4_moe_ep2_select_top3_balance_scores_fn():
    _ensure_mxfp4_moe_route_imports()
    return _MXFP4_MOE_EP2_SELECT_TOP3_BALANCE_SCORES_FN


def _get_mxfp4_moe_ep2_scatter_top3_scores_fn():
    _ensure_mxfp4_moe_route_imports()
    return _MXFP4_MOE_EP2_SCATTER_TOP3_SCORES_FN


def _get_mxfp4_moe_scatter_scores_fn():
    _ensure_mxfp4_moe_route_imports()
    return _MXFP4_MOE_SCATTER_SCORES_FN


def _get_mxfp4_moe_gather_scores_fn():
    _ensure_mxfp4_moe_route_imports()
    return _MXFP4_MOE_GATHER_SCORES_FN


def _get_mxfp4_moe_build_route_inverse_fn():
    _ensure_mxfp4_moe_route_imports()
    return _MXFP4_MOE_BUILD_ROUTE_INVERSE_FN


def _get_mxfp4_moe_route_combine_add_fn():
    _ensure_mxfp4_moe_route_imports()
    return _MXFP4_MOE_ROUTE_COMBINE_ADD_FN


def _get_mxfp4_moe_scale_scatter_add_fn():
    _ensure_mxfp4_moe_route_imports()
    return _MXFP4_MOE_SCALE_SCATTER_ADD_FN


def _get_mxfp4_moe_indexed_dot_rows_fn():
    _ensure_mxfp4_moe_route_imports()
    return _MXFP4_MOE_INDEXED_DOT_ROWS_FN


def _get_mxfp4_moe_indexed_scale_dot_rows_fn():
    _ensure_mxfp4_moe_route_imports()
    return _MXFP4_MOE_INDEXED_SCALE_DOT_ROWS_FN


def _get_mxfp4_moe_indexed_scale_rows_fn():
    _ensure_mxfp4_moe_route_imports()
    return _MXFP4_MOE_INDEXED_SCALE_ROWS_FN


def _ensure_mxfp4_moe_route_imports():
    global _MXFP4_MOE_REORDER_FN, _MXFP4_MOE_REORDER_SCORES_FULL_FN
    global _MXFP4_MOE_EP2_BALANCE_TOP3_ROUTES_FN
    global _MXFP4_MOE_EP2_SELECT_TOP3_BALANCE_SCORES_FN
    global _MXFP4_MOE_EP2_SCATTER_TOP3_SCORES_FN
    global _MXFP4_MOE_GATHER_SCORES_FN, _MXFP4_MOE_SCATTER_SCORES_FN
    global _MXFP4_MOE_BUILD_ROUTE_INVERSE_FN
    global _MXFP4_MOE_ROUTE_COMBINE_ADD_FN, _MXFP4_MOE_SCALE_SCATTER_ADD_FN
    global _MXFP4_MOE_INDEXED_DOT_ROWS_FN, _MXFP4_MOE_INDEXED_SCALE_DOT_ROWS_FN
    global _MXFP4_MOE_INDEXED_SCALE_ROWS_FN
    global _MXFP4_MOE_REORDER_IMPORT_ATTEMPTED
    if _MXFP4_MOE_REORDER_IMPORT_ATTEMPTED:
        return
    _MXFP4_MOE_REORDER_IMPORT_ATTEMPTED = True
    try:
        from low_bits_training.quantization import mxfp4_backend
    except ImportError:
        _MXFP4_MOE_REORDER_FN = None
        _MXFP4_MOE_REORDER_SCORES_FULL_FN = None
        _MXFP4_MOE_EP2_BALANCE_TOP3_ROUTES_FN = None
        _MXFP4_MOE_EP2_SELECT_TOP3_BALANCE_SCORES_FN = None
        _MXFP4_MOE_EP2_SCATTER_TOP3_SCORES_FN = None
        _MXFP4_MOE_GATHER_SCORES_FN = None
        _MXFP4_MOE_SCATTER_SCORES_FN = None
        _MXFP4_MOE_BUILD_ROUTE_INVERSE_FN = None
        _MXFP4_MOE_ROUTE_COMBINE_ADD_FN = None
        _MXFP4_MOE_SCALE_SCATTER_ADD_FN = None
        _MXFP4_MOE_INDEXED_DOT_ROWS_FN = None
        _MXFP4_MOE_INDEXED_SCALE_DOT_ROWS_FN = None
        _MXFP4_MOE_INDEXED_SCALE_ROWS_FN = None
    else:
        _MXFP4_MOE_REORDER_FN = getattr(mxfp4_backend, "mxfp4_moe_reorder_indices", None)
        _MXFP4_MOE_REORDER_SCORES_FULL_FN = getattr(
            mxfp4_backend, "mxfp4_moe_reorder_scores_full", None
        )
        _MXFP4_MOE_EP2_BALANCE_TOP3_ROUTES_FN = getattr(
            mxfp4_backend, "mxfp4_moe_ep2_balance_top3_routes", None
        )
        _MXFP4_MOE_EP2_SELECT_TOP3_BALANCE_SCORES_FN = getattr(
            mxfp4_backend, "mxfp4_moe_ep2_select_top3_balance_scores", None
        )
        _MXFP4_MOE_EP2_SCATTER_TOP3_SCORES_FN = getattr(
            mxfp4_backend, "mxfp4_moe_ep2_scatter_top3_scores", None
        )
        _MXFP4_MOE_GATHER_SCORES_FN = getattr(mxfp4_backend, "mxfp4_moe_gather_scores", None)
        _MXFP4_MOE_SCATTER_SCORES_FN = getattr(mxfp4_backend, "mxfp4_moe_scatter_scores", None)
        _MXFP4_MOE_BUILD_ROUTE_INVERSE_FN = getattr(
            mxfp4_backend, "mxfp4_moe_build_route_inverse", None
        )
        _MXFP4_MOE_ROUTE_COMBINE_ADD_FN = getattr(
            mxfp4_backend, "mxfp4_moe_route_combine_add_bf16", None
        )
        _MXFP4_MOE_SCALE_SCATTER_ADD_FN = getattr(
            mxfp4_backend, "mxfp4_moe_scale_scatter_add_bf16", None
        )
        _MXFP4_MOE_INDEXED_DOT_ROWS_FN = getattr(
            mxfp4_backend, "mxfp4_moe_indexed_dot_rows_bf16", None
        )
        _MXFP4_MOE_INDEXED_SCALE_DOT_ROWS_FN = getattr(
            mxfp4_backend, "mxfp4_moe_indexed_scale_dot_rows_bf16", None
        )
        _MXFP4_MOE_INDEXED_SCALE_ROWS_FN = getattr(
            mxfp4_backend, "mxfp4_moe_indexed_scale_rows_bf16", None
        )


def _mxfp4_deepseek_grouped_m_granularity() -> int:
    raw = os.environ.get("MXFP4_DEEPSEEK_GROUPED_M_GRANULARITY", "256")
    try:
        value = int(raw)
    except ValueError:
        value = 256
    return value if value in (256, 512, 1024) else 256


def _mxfp4_deepseek_allow_unsafe_tk_ep_route() -> bool:
    return _lbt_env_flag("MXFP4_DEEPSEEK_ALLOW_UNSAFE_TK_EP_ROUTE", False)


def _mxfp4_deepseek_validate_tk_ep_route() -> bool:
    return _lbt_env_flag("MXFP4_DEEPSEEK_VALIDATE_TK_EP_ROUTE", False)


def _mxfp4_deepseek_can_use_tk_route_producer() -> bool:
    return _lbt_dist_world_size() <= 1 or _mxfp4_deepseek_allow_unsafe_tk_ep_route()


def _mxfp4_deepseek_route_scores_full_producer(num_experts: int) -> bool:
    if not _mxfp4_deepseek_can_use_tk_route_producer():
        return False
    raw = os.environ.get("MXFP4_DEEPSEEK_ROUTE_SCORES_FULL_PRODUCER", "auto").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return int(num_experts) <= 16


def _mxfp4_deepseek_route_scores_full_fallback() -> bool:
    return (
        _mxfp4_deepseek_can_use_tk_route_producer()
        and _lbt_env_flag("MXFP4_DEEPSEEK_ROUTE_SCORES_FULL_FALLBACK", False)
    )


def _mxfp4_deepseek_tk_scored_fallback_combine() -> bool:
    return _lbt_env_flag("MXFP4_DEEPSEEK_TK_SCORED_FALLBACK_COMBINE", True)


def _mxfp4_deepseek_tk_scored_fallback_combine_fwd() -> bool:
    return _lbt_env_flag(
        "MXFP4_DEEPSEEK_TK_SCORED_FALLBACK_COMBINE_FWD",
        _mxfp4_deepseek_tk_scored_fallback_combine(),
    )


def _mxfp4_deepseek_tk_scored_fallback_combine_bwd() -> bool:
    return _lbt_env_flag(
        "MXFP4_DEEPSEEK_TK_SCORED_FALLBACK_COMBINE_BWD",
        True,
    )


def _mxfp4_deepseek_tk_scored_route_inverse_combine() -> bool:
    return _lbt_env_flag("MXFP4_DEEPSEEK_TK_SCORED_ROUTE_INVERSE_COMBINE", False)


def _mxfp4_deepseek_tk_ep2_balance_routes() -> bool:
    return _lbt_env_flag("MXFP4_DEEPSEEK_TK_EP2_BALANCE_ROUTES", True)


def _mxfp4_deepseek_tk_ep2_select_top3() -> bool:
    return _lbt_env_flag("MXFP4_DEEPSEEK_TK_EP2_SELECT_TOP3", True)


def _mxfp4_deepseek_tk_indexed_scale_bwd() -> bool:
    return _lbt_env_flag("MXFP4_DEEPSEEK_TK_INDEXED_SCALE_BWD", True)


def _mxfp4_deepseek_tk_score_gather() -> bool:
    return _lbt_env_flag("MXFP4_DEEPSEEK_TK_SCORE_GATHER", True)


def _mxfp4_deepseek_tk_route_inverse_fallback_combine() -> bool:
    return _lbt_env_flag("MXFP4_DEEPSEEK_TK_ROUTE_INVERSE_FALLBACK_COMBINE", False)


def _mxfp4_deepseek_indexed_fallback_combine() -> bool:
    return _lbt_env_flag("MXFP4_DEEPSEEK_INDEXED_FALLBACK_COMBINE", True)


def _mxfp4_deepseek_scored_indexed_fallback_combine() -> bool:
    return _lbt_env_flag("MXFP4_DEEPSEEK_SCORED_INDEXED_FALLBACK_COMBINE", True)


def _mxfp4_as_contiguous_dtype(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if tensor.dtype != dtype:
        tensor = tensor.to(dtype)
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return tensor


def _lbt_wait_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.is_cuda:
        return torch.ops._c10d_functional.wait_tensor(tensor)
    return tensor


def _lbt_validate_ep_route_counts(counts: torch.Tensor, routes: int) -> None:
    if not _mxfp4_deepseek_validate_tk_ep_route():
        return
    if (
        _lbt_dist_world_size() <= 1
        and not _mxfp4_deepseek_allow_unsafe_tk_ep_route()
    ):
        return
    if int(counts.to(torch.int64).sum().item()) != int(routes):
        raise RuntimeError("TK route producer counts do not match routed rows under EP")


def _lbt_ep_route_counts_match(counts: torch.Tensor, routes: int) -> bool:
    if not _mxfp4_deepseek_validate_tk_ep_route():
        return True
    if (
        _lbt_dist_world_size() <= 1
        and not _mxfp4_deepseek_allow_unsafe_tk_ep_route()
    ):
        return True
    return int(counts.to(torch.int64).sum().item()) == int(routes)


def _lbt_route_positions_are_permutation(
    route_positions: torch.Tensor | None,
    routes: int,
) -> bool:
    if route_positions is None:
        return False
    route_positions = route_positions.reshape(-1)
    if int(route_positions.numel()) != int(routes):
        return False
    if routes == 0:
        return True
    if int(route_positions.min().item()) < 0:
        return False
    if int(route_positions.max().item()) >= int(routes):
        return False
    return int(torch.unique(route_positions).numel()) == int(routes)


def _lbt_sync_unsafe_tk_route(tensor: torch.Tensor) -> None:
    if _mxfp4_deepseek_allow_unsafe_tk_ep_route() and tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)


class _MXFP4MoERouteScoresFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
        num_experts: int,
        top_k: int,
        pad_granularity: int,
    ):
        fn = _get_mxfp4_moe_reorder_scores_full_fn()
        if fn is None:
            raise AttributeError("mxfp4_moe_reorder_scores_full unavailable")
        flat_scores = _mxfp4_as_contiguous_dtype(top_scores.reshape(-1), torch.float32)
        selected = _mxfp4_as_contiguous_dtype(selected_experts_indices, torch.int64)
        (
            sorted_scores,
            token_indices,
            counts,
            route_positions,
            route_inverse,
            route_inverse_padded,
        ) = fn(flat_scores, selected, int(num_experts), int(top_k), int(pad_granularity))
        ctx.orig_shape = tuple(top_scores.shape)
        ctx.num_scores = int(flat_scores.numel())
        ctx.save_for_backward(route_positions)
        ctx.mark_non_differentiable(token_indices, counts, route_positions, route_inverse, route_inverse_padded)
        return sorted_scores, token_indices, counts, route_positions, route_inverse, route_inverse_padded

    @staticmethod
    def backward(ctx, grad_sorted_scores, *_unused):
        (route_positions,) = ctx.saved_tensors
        scatter_fn = _get_mxfp4_moe_scatter_scores_fn()
        if scatter_fn is None:
            grad_flat = torch.empty(
                (ctx.num_scores,),
                device=grad_sorted_scores.device,
                dtype=torch.float32,
            )
            grad_flat[route_positions] = _mxfp4_as_contiguous_dtype(grad_sorted_scores, torch.float32)
        else:
            grad_flat = scatter_fn(
                _mxfp4_as_contiguous_dtype(grad_sorted_scores, torch.float32),
                route_positions,
                ctx.num_scores,
            )
        return grad_flat.reshape(ctx.orig_shape), None, None, None, None


class _MXFP4MoEScoreGatherFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores: torch.Tensor, route_positions: torch.Tensor):
        route_positions = _mxfp4_as_contiguous_dtype(route_positions.reshape(-1), torch.int64)
        flat_scores = _mxfp4_as_contiguous_dtype(scores.reshape(-1), torch.float32)
        ctx.orig_shape = tuple(scores.shape)
        ctx.score_dtype = scores.dtype
        ctx.num_scores = int(flat_scores.numel())
        ctx.save_for_backward(route_positions)
        gather_fn = _get_mxfp4_moe_gather_scores_fn()
        if gather_fn is None:
            return flat_scores[route_positions].to(ctx.score_dtype)
        try:
            return gather_fn(flat_scores, route_positions).to(ctx.score_dtype)
        except (AttributeError, FileNotFoundError, ImportError, RuntimeError):
            return flat_scores[route_positions].to(ctx.score_dtype)

    @staticmethod
    def backward(ctx, grad_sorted_scores: torch.Tensor):
        (route_positions,) = ctx.saved_tensors
        grad_sorted_scores = _mxfp4_as_contiguous_dtype(
            grad_sorted_scores.reshape(-1),
            torch.float32,
        )
        scatter_fn = _get_mxfp4_moe_scatter_scores_fn()
        if scatter_fn is None:
            grad_flat = torch.empty(
                (ctx.num_scores,),
                device=grad_sorted_scores.device,
                dtype=torch.float32,
            )
            grad_flat[route_positions] = grad_sorted_scores
        else:
            try:
                grad_flat = scatter_fn(grad_sorted_scores, route_positions, ctx.num_scores)
            except (AttributeError, FileNotFoundError, ImportError, RuntimeError):
                grad_flat = torch.empty(
                    (ctx.num_scores,),
                    device=grad_sorted_scores.device,
                    dtype=torch.float32,
                )
                grad_flat[route_positions] = grad_sorted_scores
        return grad_flat.reshape(ctx.orig_shape).to(ctx.score_dtype), None


class _MXFP4MoEEP2SelectTop3BalanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        scores: torch.Tensor,
        expert_bias: torch.Tensor | None,
        experts_per_group: int,
    ):
        fn = _get_mxfp4_moe_ep2_select_top3_balance_scores_fn()
        if fn is None:
            raise AttributeError("mxfp4_moe_ep2_select_top3_balance_scores unavailable")
        scores_f32 = _mxfp4_as_contiguous_dtype(scores, torch.float32)
        if expert_bias is None:
            bias = scores_f32.new_empty((0,))
        else:
            bias = _mxfp4_as_contiguous_dtype(expert_bias, torch.float32)
        top_scores, selected_experts_indices = fn(
            scores_f32,
            bias,
            int(experts_per_group),
        )
        ctx.orig_shape = tuple(scores.shape)
        ctx.score_dtype = scores.dtype
        ctx.save_for_backward(selected_experts_indices)
        ctx.mark_non_differentiable(selected_experts_indices)
        return top_scores.to(scores.dtype), selected_experts_indices

    @staticmethod
    def backward(ctx, grad_top_scores: torch.Tensor, _grad_selected=None):
        (selected_experts_indices,) = ctx.saved_tensors
        grad_scores = None
        if ctx.needs_input_grad[0]:
            grad_top_scores = _mxfp4_as_contiguous_dtype(grad_top_scores, torch.float32)
            scatter_fn = _get_mxfp4_moe_ep2_scatter_top3_scores_fn()
            if scatter_fn is not None:
                try:
                    grad_scores = scatter_fn(
                        grad_top_scores,
                        selected_experts_indices,
                        int(ctx.orig_shape[1]),
                    )
                except (AttributeError, FileNotFoundError, ImportError, RuntimeError):
                    grad_scores = None
            if grad_scores is None:
                grad_scores = torch.zeros(
                    ctx.orig_shape,
                    device=grad_top_scores.device,
                    dtype=torch.float32,
                )
                grad_scores.scatter_add_(
                    1,
                    selected_experts_indices,
                    grad_top_scores,
                )
            grad_scores = grad_scores.to(ctx.score_dtype)
        return grad_scores, None, None


def _mxfp4_gather_route_scores(
    top_scores: torch.Tensor,
    route_positions: torch.Tensor,
) -> torch.Tensor:
    if (
        _mxfp4_deepseek_tk_score_gather()
        and top_scores.is_cuda
        and route_positions.is_cuda
    ):
        return _MXFP4MoEScoreGatherFunction.apply(top_scores, route_positions)
    return top_scores.reshape(-1)[route_positions]


class _MoEIndexCombineFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        base: torch.Tensor,
        token_indices: torch.Tensor,
        routed_output: torch.Tensor,
    ):
        token_indices = token_indices.reshape(-1).contiguous()
        ctx.save_for_backward(token_indices)
        out = base.clone()
        out.index_add_(0, token_indices, routed_output)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (token_indices,) = ctx.saved_tensors
        grad_base = grad_output
        grad_routed_output = grad_output.index_select(0, token_indices)
        return grad_base, None, grad_routed_output


class _MoEScoredIndexCombineFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        base: torch.Tensor,
        token_indices: torch.Tensor,
        routed_output: torch.Tensor,
        scores: torch.Tensor,
        route_inverse: torch.Tensor | None,
        top_k: int,
    ):
        token_indices = token_indices.reshape(-1).contiguous()
        score_dtype = scores.dtype
        scores = _mxfp4_as_contiguous_dtype(scores.reshape(-1), torch.float32)
        ctx.score_dtype = score_dtype
        saved_for_backward = False

        if (
            _mxfp4_deepseek_tk_scored_fallback_combine_fwd()
            and base.is_cuda
            and routed_output.is_cuda
            and token_indices.is_cuda
            and scores.is_cuda
            and base.dtype == torch.bfloat16
            and routed_output.dtype == torch.bfloat16
        ):
            routed_output = _lbt_wait_tensor(routed_output)
            ctx.save_for_backward(token_indices, routed_output, scores)
            saved_for_backward = True
            routed_output_bf16 = _mxfp4_as_contiguous_dtype(routed_output, torch.bfloat16)
            base_bf16 = _mxfp4_as_contiguous_dtype(base, torch.bfloat16)
            if (
                route_inverse is not None
                and int(top_k) > 0
                and _mxfp4_deepseek_tk_scored_route_inverse_combine()
            ):
                combine_add_fn = _get_mxfp4_moe_route_combine_add_fn()
                if combine_add_fn is not None:
                    try:
                        out = torch.empty_like(base_bf16)
                        combine_add_fn(
                            routed_output_bf16,
                            scores,
                            _mxfp4_as_contiguous_dtype(route_inverse.reshape(-1), torch.int64),
                            base_bf16,
                            out,
                            int(top_k),
                        )
                        return out
                    except (AttributeError, FileNotFoundError, ImportError):
                        pass
            scale_scatter_fn = _get_mxfp4_moe_scale_scatter_add_fn()
            if scale_scatter_fn is not None:
                try:
                    out = base_bf16.clone()
                    scale_scatter_fn(routed_output_bf16, scores, token_indices, out)
                    return out
                except (AttributeError, FileNotFoundError, ImportError):
                    pass

        if not saved_for_backward:
            ctx.save_for_backward(token_indices, routed_output, scores)
        scaled_output = (
            routed_output.to(torch.float32)
            * scores.reshape(-1, 1)
        ).to(base.dtype)
        out = base.clone()
        out.index_add_(0, token_indices, scaled_output)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        token_indices, routed_output, scores = ctx.saved_tensors
        grad_base = grad_output

        if (
            _mxfp4_deepseek_tk_scored_fallback_combine_bwd()
            and grad_output.is_cuda
            and routed_output.is_cuda
            and token_indices.is_cuda
            and scores.is_cuda
            and grad_output.dtype == torch.bfloat16
            and routed_output.dtype == torch.bfloat16
        ):
            grad_output = _lbt_wait_tensor(grad_output)
            routed_output = _lbt_wait_tensor(routed_output)
            fused_bwd_fn = _get_mxfp4_moe_indexed_scale_dot_rows_fn()
            if fused_bwd_fn is not None:
                try:
                    grad_routed_output, grad_scores = fused_bwd_fn(
                        _mxfp4_as_contiguous_dtype(grad_output, torch.bfloat16),
                        token_indices,
                        _mxfp4_as_contiguous_dtype(routed_output, torch.bfloat16),
                        scores,
                    )
                    return (
                        grad_base,
                        None,
                        grad_routed_output,
                        grad_scores.to(ctx.score_dtype),
                        None,
                        None,
                    )
                except (AttributeError, FileNotFoundError, ImportError):
                    pass

        grad_selected = None
        grad_selected_fp32 = None
        grad_output_bf16 = None
        routed_output_fp32 = None
        grad_routed_output = None
        scale_rows_fn = _get_mxfp4_moe_indexed_scale_rows_fn()

        def _materialize_grad_output_bf16() -> torch.Tensor:
            nonlocal grad_output_bf16
            if grad_output_bf16 is None:
                grad_output_bf16 = _mxfp4_as_contiguous_dtype(grad_output, torch.bfloat16)
            return grad_output_bf16

        if (
            _mxfp4_deepseek_tk_indexed_scale_bwd()
            and scale_rows_fn is not None
            and grad_output.is_cuda
            and token_indices.is_cuda
            and scores.is_cuda
            and grad_output.dtype == torch.bfloat16
            and routed_output.dtype == torch.bfloat16
        ):
            try:
                grad_routed_output = scale_rows_fn(
                    _materialize_grad_output_bf16(),
                    token_indices,
                    scores,
                )
            except (AttributeError, FileNotFoundError, ImportError, RuntimeError):
                grad_routed_output = None

        def _materialize_grad_selected_fp32() -> torch.Tensor:
            nonlocal grad_selected, grad_selected_fp32
            if grad_selected_fp32 is None:
                grad_selected = grad_output.index_select(0, token_indices)
                grad_selected_fp32 = grad_selected.to(torch.float32)
            return grad_selected_fp32

        if grad_routed_output is None:
            grad_routed_output = (
                _materialize_grad_selected_fp32() * scores.reshape(-1, 1)
            ).to(routed_output.dtype)

        def _materialize_routed_output_fp32() -> torch.Tensor:
            nonlocal routed_output_fp32
            if routed_output_fp32 is None:
                routed_output_fp32 = routed_output.to(torch.float32)
            return routed_output_fp32

        dot_fn = _get_mxfp4_moe_indexed_dot_rows_fn()
        if (
            dot_fn is not None
            and grad_output.is_cuda
            and routed_output.is_cuda
            and grad_output.dtype == torch.bfloat16
            and routed_output.dtype == torch.bfloat16
        ):
            try:
                routed_output = _lbt_wait_tensor(routed_output)
                grad_scores = dot_fn(
                    _materialize_grad_output_bf16(),
                    token_indices,
                    _mxfp4_as_contiguous_dtype(routed_output, torch.bfloat16),
                )
            except (AttributeError, FileNotFoundError, ImportError, RuntimeError):
                grad_scores = (
                    _materialize_grad_selected_fp32()
                    * _materialize_routed_output_fp32()
                ).sum(dim=1)
        else:
            grad_scores = (
                _materialize_grad_selected_fp32()
                * _materialize_routed_output_fp32()
            ).sum(dim=1)
        return (
            grad_base,
            None,
            grad_routed_output,
            grad_scores.to(ctx.score_dtype),
            None,
            None,
        )


@dataclass
class MoEArgs:
    num_experts: int = 8
    num_shared_experts: int = 1

    # router
    score_func: Literal["softmax", "sigmoid"] = "sigmoid"
    route_norm: bool = False
    route_scale: float = 1.0
    score_before_experts: bool = True

    # token-choice
    top_k: int = 1
    use_grouped_mm: bool = True  # grouped mm or for-loop for the experts computation
    load_balance_coeff: float | None = 1e-3

    _debug_force_load_balance: bool = False
    # if True, we force each experts get same amount of token via round-robin


# can be used as dense FFN layer or shared experts in MoE layers
class FeedForward(nn.Module):
    """
    Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden dimension of the feedforward layer.

    Attributes:
        w1 (Linear): Linear transformation for the first layer.
        w2 (Linear): Linear transformation for the second layer.
        w3 (Linear): Linear transformation for the third layer.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def init_weights(self, init_std: float = 0.02):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


# NOTE: keeping this for-loop implementation for comparison
#       and readability, may remove later
def _run_experts_for_loop(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    # NOTE: this would incur a synchronization between device and host
    num_tokens_per_expert = num_tokens_per_expert.tolist()

    # side-effect code due to the usage of generate_permute_indices
    num_padding = x.shape[0] - sum(num_tokens_per_expert)

    # a tuple of tensors indexed by experts
    # each with shape (tokens_per_expert(varying), dim)
    x = torch.split(
        x[: sum(num_tokens_per_expert)],
        split_size_or_sections=num_tokens_per_expert,
        dim=0,
    )
    out_experts_splits = []
    for expert_idx, x_expert in enumerate(x):
        h = F.silu(torch.matmul(x_expert, w1[expert_idx].transpose(-2, -1)))
        h = h * torch.matmul(x_expert, w3[expert_idx].transpose(-2, -1))
        h = torch.matmul(h, w2[expert_idx].transpose(-2, -1))
        # h shape (tokens_per_expert(varying), dim)
        out_experts_splits.append(h)
    out = torch.cat(out_experts_splits, dim=0)

    # side-effect code due to the usage of generate_permute_indices
    out = torch.vstack((out, out.new_zeros((num_padding, out.shape[-1]))))

    return out


def _run_experts_grouped_mm(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)

    h = F.silu(
        torch._grouped_mm(x.bfloat16(), w1.bfloat16().transpose(-2, -1), offs=offsets)
    )
    h = h * torch._grouped_mm(
        x.bfloat16(), w3.bfloat16().transpose(-2, -1), offs=offsets
    )
    out = torch._grouped_mm(h, w2.bfloat16().transpose(-2, -1), offs=offsets).type_as(x)

    return out


class GroupedExperts(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int,
        use_grouped_mm: bool,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.w1 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.w2 = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        self.w3 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.use_grouped_mm = use_grouped_mm

    def forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(self.w1, DTensor):
            # Convert parameters from DTensors to plain Tensors, to work with
            # dynamic-shape inputs in EP which cannot be easily expressed as DTensors.
            w1 = self.w1.to_local()
            w2 = self.w2.to_local()
            w3 = self.w3.to_local()
        else:
            w1 = self.w1
            w2 = self.w2
            w3 = self.w3

        if self.use_grouped_mm:
            # NOTE: If EP is not used, we need to pad the indices
            #       to prepare for grouped_mm;
            #       otherwise, EP will handle the padding.
            if (
                not isinstance(self.w1, DTensor)
                or "ep" not in self.w1.device_mesh.mesh_dim_names
            ):
                run_experts_fn = indices_padding_wrapper(_run_experts_grouped_mm)
            else:
                run_experts_fn = _run_experts_grouped_mm
            return run_experts_fn(w1, w2, w3, x, num_tokens_per_expert)
        else:
            return _run_experts_for_loop(w1, w2, w3, x, num_tokens_per_expert)

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w2, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.w3, mean=0.0, std=init_std)


class TokenChoiceTopKRouter(nn.Module):
    """This class implements token-choice routing. In token-choice top-K routing, each token is
        routed to top K experts based on the router scores.

    Args:
        dim (int): Dimension of input tokens.
        num_experts (int): Number of experts in each moe layer.
        top_k (int): Number of experts each token will be routed to in token-choice routing.
        score_func (Literal["softmax", "sigmoid"]): Whether to use sigmoid or softmax for router scores.
        route_norm (bool): Whether to normalize the routing scores when using sigmoid.
        route_scale (float): Scaling factor applied to the routing scores.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int,
        score_func: Literal["softmax", "sigmoid"],
        route_norm: bool,
        route_scale: float,
        _debug_force_load_balance: bool = False,
    ):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.num_experts = num_experts
        self.top_k = top_k
        self.score_func = score_func
        self.route_norm = route_norm
        self.route_scale = route_scale
        self._debug_force_load_balance = _debug_force_load_balance
        self._debug_force_load_balance_counts_cache = {}
        self._debug_force_load_balance_indices_cache = {}
        self._ep_route_coverage_stripe_cache = {}

    def _debug_force_load_balance_routing(
        self, scores: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Balanced round-robin expert assignment.
        Returns (selected_experts_indices [N, K] LongTensor, top_scores [N, K] FloatTensor).
        """
        n_tokens = scores.size(0)
        cache_key = (
            n_tokens * self.top_k,
            scores.device.type,
            scores.device.index,
        )
        selected_experts_indices = self._debug_force_load_balance_indices_cache.get(cache_key)
        if selected_experts_indices is None:
            # Round-robin indices with exact balance
            selected_experts_indices = (
                torch.arange(
                    n_tokens * self.top_k, device=scores.device, dtype=torch.int64
                ).reshape(n_tokens, self.top_k)
                % self.num_experts
            )
            self._debug_force_load_balance_indices_cache[cache_key] = selected_experts_indices
        top_scores = scores.gather(dim=1, index=selected_experts_indices)  # [N,K]
        return selected_experts_indices, top_scores

    def _debug_force_load_balance_counts(
        self,
        n_tokens: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_routes = n_tokens * self.top_k
        cache_key = (total_routes, device.type, device.index)
        cached = self._debug_force_load_balance_counts_cache.get(cache_key)
        if cached is not None:
            return cached
        base = total_routes // self.num_experts
        rem = total_routes - base * self.num_experts
        counts = torch.full(
            (self.num_experts,),
            float(base),
            device=device,
            dtype=torch.float32,
        )
        if rem > 0:
            counts[:rem] += 1.0
        self._debug_force_load_balance_counts_cache[cache_key] = counts
        return counts

    def _ep_route_coverage_degree(self) -> int:
        ep_degree = _lbt_env_int("LBT_MOE_EP_ROUTE_COVERAGE_DEGREE", 0)
        if (
            _lbt_env_flag("LBT_MOE_EP_ROUTE_COVERAGE")
            and ep_degree == 2
            and self.top_k >= ep_degree
            and self.num_experts % ep_degree == 0
        ):
            return ep_degree
        return 0

    def _ep_route_coverage_mode(self) -> str:
        return os.environ.get("LBT_MOE_EP_ROUTE_COVERAGE_MODE", "best").strip().lower()

    def _use_striped_ep_route_coverage(self) -> bool:
        mode = self._ep_route_coverage_mode()
        return mode in ("stripe", "striped", "balanced") and self.top_k == 3

    def _ep_route_stripe_tensors(
        self,
        n_tokens: int,
        experts_per_group: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cache_key = (n_tokens, experts_per_group, device.type, device.index)
        cached = self._ep_route_coverage_stripe_cache.get(cache_key)
        if cached is not None:
            return cached
        indices = torch.arange(n_tokens, device=device, dtype=torch.int64)
        local_indices = indices.remainder(experts_per_group)
        remote_indices = local_indices + experts_per_group
        group1_mask = (indices & 1).bool()
        cached = (local_indices, remote_indices, group1_mask)
        self._ep_route_coverage_stripe_cache[cache_key] = cached
        return cached

    def _select_striped_ep_route_coverage(
        self,
        route_scores: torch.Tensor,
    ) -> torch.Tensor:
        experts_per_group = self.num_experts // 2
        group0 = torch.topk(route_scores[:, :experts_per_group], k=2, dim=1).indices
        group1 = (
            torch.topk(route_scores[:, experts_per_group:], k=2, dim=1).indices
            + experts_per_group
        )
        group0_major = torch.stack((group0[:, 0], group0[:, 1], group1[:, 0]), dim=1)
        group1_major = torch.stack((group1[:, 0], group1[:, 1], group0[:, 0]), dim=1)
        _stripe_local, _stripe_remote, group1_mask = self._ep_route_stripe_tensors(
            int(route_scores.shape[0]),
            experts_per_group,
            route_scores.device,
        )
        return torch.where(group1_mask.unsqueeze(1), group1_major, group0_major)

    def _apply_striped_balance_ep_route_coverage(
        self,
        selected_experts_indices: torch.Tensor,
        selected_group1: torch.Tensor,
        experts_per_group: int,
    ) -> torch.Tensor:
        if (
            _mxfp4_deepseek_tk_ep2_balance_routes()
            and self.top_k == 3
            and selected_experts_indices.is_cuda
            and selected_experts_indices.dtype == torch.int64
        ):
            balance_fn = _get_mxfp4_moe_ep2_balance_top3_routes_fn()
            if balance_fn is not None:
                try:
                    return balance_fn(
                        selected_experts_indices.contiguous(),
                        experts_per_group,
                    )
                except (AttributeError, FileNotFoundError, ImportError, RuntimeError):
                    pass

        base_local, base_remote, group1_mask = self._ep_route_stripe_tensors(
            int(selected_experts_indices.shape[0]),
            experts_per_group,
            selected_experts_indices.device,
        )
        group0_candidate = base_local
        group1_candidate = base_remote
        group0_candidate = torch.where(
            (selected_experts_indices == group0_candidate.unsqueeze(1)).any(dim=1),
            (base_local + 1).remainder(experts_per_group),
            group0_candidate,
        )
        group1_candidate = torch.where(
            (selected_experts_indices == group1_candidate.unsqueeze(1)).any(dim=1),
            (base_local + 1).remainder(experts_per_group) + experts_per_group,
            group1_candidate,
        )
        group0_candidate_2 = (group0_candidate + 1).remainder(experts_per_group)
        group1_candidate_2 = (
            (group1_candidate - experts_per_group + 1).remainder(experts_per_group)
            + experts_per_group
        )

        current_group1 = selected_group1.to(torch.int64).sum(dim=1)
        desired_group1 = group1_mask.to(current_group1.dtype) + 1
        need_group1 = torch.clamp_min(desired_group1 - current_group1, 0)
        need_group0 = torch.clamp_min(current_group1 - desired_group1, 0)
        selected_experts_indices = selected_experts_indices.clone()
        added_group1 = torch.zeros_like(need_group1)
        added_group0 = torch.zeros_like(need_group0)

        for pos in range(self.top_k - 1, -1, -1):
            replace_group1 = (~selected_group1[:, pos]) & (need_group1 > 0)
            replacement_group1 = torch.where(
                added_group1 == 0,
                group1_candidate,
                group1_candidate_2,
            )
            selected_experts_indices[:, pos] = torch.where(
                replace_group1,
                replacement_group1,
                selected_experts_indices[:, pos],
            )
            added_group1 = added_group1 + replace_group1.to(added_group1.dtype)
            need_group1 = need_group1 - replace_group1.to(need_group1.dtype)

            replace_group0 = selected_group1[:, pos] & (need_group0 > 0)
            replacement_group0 = torch.where(
                added_group0 == 0,
                group0_candidate,
                group0_candidate_2,
            )
            selected_experts_indices[:, pos] = torch.where(
                replace_group0,
                replacement_group0,
                selected_experts_indices[:, pos],
            )
            added_group0 = added_group0 + replace_group0.to(added_group0.dtype)
            need_group0 = need_group0 - replace_group0.to(need_group0.dtype)

        return selected_experts_indices

    def _apply_ep_route_coverage(
        self,
        route_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
    ) -> torch.Tensor:
        ep_degree = self._ep_route_coverage_degree()
        if ep_degree != 2:
            return selected_experts_indices

        experts_per_group = self.num_experts // ep_degree
        selected_group1 = selected_experts_indices >= experts_per_group

        if self._ep_route_coverage_mode() in ("balanced_fast", "fast_balanced"):
            return self._apply_striped_balance_ep_route_coverage(
                selected_experts_indices,
                selected_group1,
                experts_per_group,
            )

        missing_group0 = selected_group1.all(dim=1)
        missing_group1 = (~selected_group1).all(dim=1)
        selected_experts_indices = selected_experts_indices.clone()
        if self._ep_route_coverage_mode() in ("fast", "striped_missing", "round_robin"):
            base_local, base_remote, _group1_mask = self._ep_route_stripe_tensors(
                int(route_scores.shape[0]),
                experts_per_group,
                route_scores.device,
            )
            best_group0 = base_local
            best_group1 = base_remote
        else:
            best_group0 = torch.argmax(route_scores[:, :experts_per_group], dim=1)
            best_group1 = (
                torch.argmax(route_scores[:, experts_per_group:], dim=1)
                + experts_per_group
            )
        replacement = torch.where(
            missing_group0,
            best_group0,
            selected_experts_indices[:, -1],
        )
        replacement = torch.where(missing_group1, best_group1, replacement)
        selected_experts_indices[:, -1] = replacement
        return selected_experts_indices

    def _try_mxfp4_ep2_select_top3_balance(
        self,
        scores: torch.Tensor,
        expert_bias: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        bias = _lbt_local_tensor(expert_bias) if expert_bias is not None else None
        if (
            not _mxfp4_deepseek_tk_ep2_select_top3()
            or self.top_k != 3
            or self._ep_route_coverage_degree() != 2
            or self._ep_route_coverage_mode() not in ("balanced_fast", "fast_balanced")
            or not scores.is_cuda
            or scores.dtype != torch.float32
            or scores.dim() != 2
            or self.num_experts % 2 != 0
            or (
                bias is not None
                and (
                    not bias.is_cuda
                    or bias.device != scores.device
                    or bias.dim() != 1
                    or bias.numel() != self.num_experts
                )
            )
        ):
            return None
        if _get_mxfp4_moe_ep2_select_top3_balance_scores_fn() is None:
            return None
        try:
            return _MXFP4MoEEP2SelectTop3BalanceFunction.apply(
                scores,
                bias,
                self.num_experts // 2,
            )
        except (AttributeError, FileNotFoundError, ImportError, RuntimeError):
            return None

    def forward(
        self, x: torch.Tensor, expert_bias: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs*slen, dim)``.
            expert_bias (torch.Tensor | None, optional): Optional bias tensor for experts with shape ``(num_experts,)``.
                Used for load balancing. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - top_scores (torch.Tensor):
                    Routing scores for selected experts with shape ``(bs*slen, top_k)``.
                - selected_experts_indices (torch.Tensor):
                    Expert indices selected for each token with shape ``(bs*slen, top_k)``.
                - num_tokens_per_expert (torch.Tensor):
                    Number of tokens assigned to each expert with shape ``(num_experts,)``.
        """
        # scores shape (bs*slen, num_experts)
        scores = self.gate(x)

        # By default, sigmoid or softmax is performed in float32 to avoid loss explosion
        if self.score_func == "sigmoid":
            scores = torch.sigmoid(scores.to(torch.float32))
        elif self.score_func == "softmax":
            scores = F.softmax(scores.to(torch.float32), dim=1)
        else:
            raise NotImplementedError(f"Unknown score function {self.score_func}")

        if self._debug_force_load_balance:
            (
                selected_experts_indices,
                top_scores,
            ) = self._debug_force_load_balance_routing(scores)
        # top scores shape (bs*slen, top_k)
        # NOTE: The expert_bias is only used for routing. The gating value
        #       top_scores is still derived from the original scores.
        elif expert_bias is not None:
            fused_route = self._try_mxfp4_ep2_select_top3_balance(scores, expert_bias)
            if fused_route is not None:
                top_scores, selected_experts_indices = fused_route
            else:
                route_scores = scores + expert_bias
                if self._ep_route_coverage_degree() == 2 and self._use_striped_ep_route_coverage():
                    selected_experts_indices = self._select_striped_ep_route_coverage(route_scores)
                else:
                    _, selected_experts_indices = torch.topk(
                        route_scores, k=self.top_k, dim=1
                    )
                    selected_experts_indices = self._apply_ep_route_coverage(
                        route_scores,
                        selected_experts_indices,
                    )
                top_scores = scores.gather(dim=1, index=selected_experts_indices)
        else:
            if self._ep_route_coverage_degree() == 2 and self._use_striped_ep_route_coverage():
                selected_experts_indices = self._select_striped_ep_route_coverage(scores)
                top_scores = scores.gather(dim=1, index=selected_experts_indices)
            else:
                fused_route = self._try_mxfp4_ep2_select_top3_balance(scores, None)
                if fused_route is not None:
                    top_scores, selected_experts_indices = fused_route
                else:
                    top_scores, selected_experts_indices = torch.topk(
                        scores, k=self.top_k, dim=1
                    )
                    covered_selected_experts_indices = self._apply_ep_route_coverage(
                        scores,
                        selected_experts_indices,
                    )
                    if covered_selected_experts_indices is not selected_experts_indices:
                        selected_experts_indices = covered_selected_experts_indices
                        top_scores = scores.gather(dim=1, index=selected_experts_indices)

        if "LBT_MOE_ROUTER_DEBUG" in os.environ:
            _lbt_moe_router_debug(
                self,
                x,
                scores,
                top_scores,
                selected_experts_indices,
                expert_bias,
            )

        if self.route_norm:
            denominator = top_scores.sum(dim=-1, keepdim=True) + 1e-20
            top_scores = top_scores / denominator
        if self.route_scale != 1.0:
            top_scores = top_scores * self.route_scale

        # group tokens together by expert indices from 0 to num_experts and pass that to experts forward
        if self._debug_force_load_balance:
            num_tokens_per_expert = self._debug_force_load_balance_counts(
                scores.size(0),
                scores.device,
            )
        elif getattr(self, "_mxfp4_skip_router_histc", False):
            num_tokens_per_expert = torch.empty(0, device=scores.device, dtype=torch.float32)
        else:
            num_tokens_per_expert = torch.histc(
                selected_experts_indices.view(-1),
                bins=self.num_experts,
                min=0,
                max=self.num_experts,
            )

        return top_scores, selected_experts_indices, num_tokens_per_expert

    def init_weights(self, init_std: float):
        _lbt_trunc_normal_with_dtensor_fallback_(
            self.gate.weight,
            mean=0.0,
            std=init_std,
        )


# NOTE: the reason we make this a stateless module is to support
#       expert_tensor_parallel_degree=1 with consistent TP/EP APIs.
class TokenReorderer(nn.Module):
    """
    This module reorders token indices to match the order of experts, enabling
    efficient parallel processing of tokens by experts.

    Args:
        num_experts (int): Number of experts in the MoE layer.
        top_k (int): Number of experts each token will be routed to.
    """

    def __init__(self, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self._debug_force_load_balance_cache = {}

    def forward(
        self,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reorders token indices to match the order of experts for MoE routing.

        Args:
            top_scores (torch.Tensor): Routing scores for selected experts,
                shape (batch_size * seq_len, top_k)
            selected_experts_indices (torch.Tensor): Expert indices selected for each token,
                shape (batch_size*seq_len, top_k)

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - top_scores_experts_sorted: Scores reordered to match expert ordering
                - token_indices_experts_sorted: Token indices reordered to match expert ordering
                - num_tokens_per_expert: Number of tokens assigned to each expert
        """
        # group tokens together by expert indices from 0 to num_experts and pass that to experts forward
        num_tokens_per_expert = torch.histc(
            selected_experts_indices.view(-1),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        )

        # Reorder the token indices to match the order of the experts
        # token_indices_experts_sorted shape (bs*slen*top_k,)
        route_positions_experts_sorted = torch.argsort(
            selected_experts_indices.view(-1), stable=True
        )

        top_scores_experts_sorted = _mxfp4_gather_route_scores(
            top_scores,
            route_positions_experts_sorted,
        )
        token_indices_experts_sorted = route_positions_experts_sorted // self.top_k

        return (
            top_scores_experts_sorted,
            token_indices_experts_sorted,
            num_tokens_per_expert,
        )

    def forward_with_route_positions_no_scores(
        self,
        selected_experts_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            os.environ.get("MXFP4_DEEPSEEK_TK_ROUTE_REORDER", "1") != "0"
            and _mxfp4_deepseek_can_use_tk_route_producer()
            and selected_experts_indices.is_cuda
            and selected_experts_indices.dtype == torch.int64
        ):
            reorder_fn = _get_mxfp4_moe_reorder_fn()
            if reorder_fn is not None:
                try:
                    out = reorder_fn(
                        selected_experts_indices.contiguous(),
                        self.num_experts,
                        self.top_k,
                    )
                    routes = int(out[0].numel())
                    _lbt_validate_ep_route_counts(out[1], routes)
                    if (
                        _mxfp4_deepseek_allow_unsafe_tk_ep_route()
                        and _mxfp4_deepseek_validate_tk_ep_route()
                        and not _lbt_route_positions_are_permutation(out[2], routes)
                    ):
                        raise RuntimeError("TK route producer positions are unsafe under EP")
                    return out
                except (AttributeError, FileNotFoundError, ImportError, RuntimeError):
                    _lbt_sync_unsafe_tk_route(selected_experts_indices)
                    pass

        return self.forward_with_route_positions_no_scores_torch(selected_experts_indices)

    def forward_with_route_positions_no_scores_torch(
        self,
        selected_experts_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        selected_experts_indices_flat = selected_experts_indices.reshape(-1)
        num_tokens_per_expert = torch.histc(
            selected_experts_indices_flat,
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        )

        route_positions_experts_sorted = torch.argsort(
            selected_experts_indices_flat, stable=True
        )
        token_indices_experts_sorted = route_positions_experts_sorted // self.top_k

        return (
            token_indices_experts_sorted,
            num_tokens_per_expert,
            route_positions_experts_sorted,
        )

    def forward_with_route_positions(
        self,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            token_indices_experts_sorted,
            num_tokens_per_expert,
            route_positions_experts_sorted,
        ) = self.forward_with_route_positions_no_scores(selected_experts_indices)
        top_scores_experts_sorted = _mxfp4_gather_route_scores(
            top_scores,
            route_positions_experts_sorted,
        )

        return (
            top_scores_experts_sorted,
            token_indices_experts_sorted,
            num_tokens_per_expert,
            route_positions_experts_sorted,
        )

    def forward_debug_force_load_balance_no_scores(
        self,
        top_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        total_routes = top_scores.numel()
        cache_key = (
            total_routes,
            top_scores.device.type,
            top_scores.device.index,
        )
        cached = self._debug_force_load_balance_cache.get(cache_key)
        if cached is None:
            if total_routes % self.num_experts == 0:
                route_positions_experts_sorted = (
                    torch.arange(
                        total_routes,
                        device=top_scores.device,
                        dtype=torch.int64,
                    )
                    .view(-1, self.num_experts)
                    .t()
                    .contiguous()
                    .view(-1)
                )
            else:
                rows_per_expert = (total_routes + self.num_experts - 1) // self.num_experts
                route_grid = (
                    torch.arange(self.num_experts, device=top_scores.device, dtype=torch.int64).view(-1, 1)
                    + self.num_experts
                    * torch.arange(rows_per_expert, device=top_scores.device, dtype=torch.int64).view(1, -1)
                )
                route_positions_experts_sorted = route_grid[route_grid < total_routes].contiguous()
            token_indices_experts_sorted = route_positions_experts_sorted // self.top_k
            base = total_routes // self.num_experts
            rem = total_routes - base * self.num_experts
            num_tokens_per_expert = torch.full(
                (self.num_experts,),
                float(base),
                device=top_scores.device,
                dtype=torch.float32,
            )
            if rem > 0:
                num_tokens_per_expert[:rem] += 1.0
            cached = (
                route_positions_experts_sorted,
                token_indices_experts_sorted,
                num_tokens_per_expert,
            )
            self._debug_force_load_balance_cache[cache_key] = cached
        (
            route_positions_experts_sorted,
            token_indices_experts_sorted,
            num_tokens_per_expert,
        ) = cached

        return (
            token_indices_experts_sorted,
            num_tokens_per_expert,
            route_positions_experts_sorted,
        )

    def forward_debug_force_load_balance(
        self,
        top_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            token_indices_experts_sorted,
            num_tokens_per_expert,
            route_positions_experts_sorted,
        ) = self.forward_debug_force_load_balance_no_scores(top_scores)

        top_scores_experts_sorted = _mxfp4_gather_route_scores(
            top_scores,
            route_positions_experts_sorted,
        )

        return (
            top_scores_experts_sorted,
            token_indices_experts_sorted,
            num_tokens_per_expert,
            route_positions_experts_sorted,
        )


class MoE(nn.Module):
    def __init__(self, moe_args: MoEArgs, dim: int, hidden_dim: int):
        super().__init__()

        num_experts = moe_args.num_experts
        self.experts = GroupedExperts(
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            use_grouped_mm=moe_args.use_grouped_mm,
        )
        self.router = TokenChoiceTopKRouter(
            dim=dim,
            num_experts=num_experts,
            top_k=moe_args.top_k,
            score_func=moe_args.score_func,
            route_norm=moe_args.route_norm,
            route_scale=moe_args.route_scale,
            _debug_force_load_balance=moe_args._debug_force_load_balance,
        )
        self.reorderer = TokenReorderer(num_experts=num_experts, top_k=moe_args.top_k)
        self.shared_experts = (
            FeedForward(dim=dim, hidden_dim=hidden_dim * moe_args.num_shared_experts)
            if moe_args.num_shared_experts > 0
            else None
        )
        self.score_before_experts = moe_args.score_before_experts

        # define fields for auxiliary-loss-free load balancing (https://arxiv.org/abs/2408.15664)
        # NOTE: tokens_per_expert is accumulated in the model forward pass.
        #       expert_bias is updated outside the model in an optimizer step pre hook
        #       to work with gradient accumulation.
        self.load_balance_coeff = moe_args.load_balance_coeff
        if self.load_balance_coeff is not None:
            assert self.load_balance_coeff > 0.0
            self.register_buffer(
                "expert_bias",
                torch.zeros(num_experts, dtype=torch.float32),
                persistent=True,
            )
        else:
            self.expert_bias = None
        # tokens_per_expert will be used to track expert usage and to update the expert bias for load balancing
        self.register_buffer(
            "tokens_per_expert",
            torch.zeros(num_experts, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs, slen, dim)``.

        Returns:
            out (torch.Tensor): Output tensor with shape ``(bs, slen, dim)``.
        """
        bs, slen, dim = x.shape
        x_raw = x.view(-1, dim)
        x = x_raw
        indexed_x_rms_base = None
        indexed_x_rms_weight = None
        indexed_x_rms_inv = None
        rms_norm = getattr(self, "_mxfp4_ffn_norm", None)
        rmsnorm_to_bf16 = getattr(self, "_mxfp4_rmsnorm_to_bf16", None)
        norm_weight = None
        if rms_norm is not None and rmsnorm_to_bf16 is not None:
            norm_weight = getattr(rms_norm, "weight", None)
            if norm_weight is not None:
                x, indexed_x_rms_inv = rmsnorm_to_bf16(
                    x_raw,
                    norm_weight,
                    float(getattr(rms_norm, "eps", 1e-5)),
                )
                indexed_x_rms_base = x_raw
                indexed_x_rms_weight = norm_weight

        forward_debug_enabled = "LBT_MOE_FORWARD_DEBUG" in os.environ
        if forward_debug_enabled:
            _lbt_moe_forward_debug(
                self,
                "pre_router",
                x_raw=x_raw,
                x_normed=x,
                norm_weight=norm_weight,
                inv_rms=indexed_x_rms_inv,
            )

        fused_moe_combine = (
            getattr(self.experts, "forward_moe_combine", None)
            if not self.score_before_experts
            else None
        )
        if fused_moe_combine is not None and _lbt_dist_world_size() > 1:
            can_fuse_moe_combine = getattr(self.experts, "can_fuse_moe_combine", None)
            if can_fuse_moe_combine is not None:
                try:
                    if not can_fuse_moe_combine(self.reorderer.num_experts):
                        fused_moe_combine = None
                except Exception:
                    fused_moe_combine = None
        skip_router_histc = (
            (fused_moe_combine is not None or _mxfp4_deepseek_route_scores_full_fallback())
            and os.environ.get("MXFP4_DEEPSEEK_SKIP_ROUTER_HISTC", "1") != "0"
            and not self.router._debug_force_load_balance
        )
        old_skip_router_histc = getattr(self.router, "_mxfp4_skip_router_histc", False)
        self.router._mxfp4_skip_router_histc = skip_router_histc
        try:
            # top_scores and selected_experts_indices shape (bs*slen*top_k,)
            # num_tokens_per_expert shape (num_experts,)
            (
                top_scores,
                selected_experts_indices,
                num_tokens_per_expert,
            ) = self.router(x, self.expert_bias)
        finally:
            self.router._mxfp4_skip_router_histc = old_skip_router_histc

        # tokens_per_expert will be used to update the expert bias for load balancing.
        # and also to count the expert usage
        # TODO: Activation Checkpointing has the side effect of double counting tokens_per_expert --
        #       first in the forward pass, and then in the backward pass. However, this has no
        #       effect on the expert bias update thanks to the torch.sign() operator.
        if not skip_router_histc:
            with torch.no_grad():
                self.tokens_per_expert.add_(num_tokens_per_expert)

        # top_scores and token_indices_experts_sorted shape (bs*slen*top_k,)
        # num_tokens_per_expert shape (num_experts,)
        # NOTE: the reason we need to compute num_tokens_per_expert again is:
        #       1st computation in router is to update self.tokens_per_expert
        #       which would be the same across all TP ranks.
        #       2nd computation in reorderer is for the actual routing and experts computation
        #       which would be sharded over TP ranks if expert_tensor_parallel_degree==1.
        #       If tensor_paralllel_degree == expert_tensor_parallel_degree, they agree.
        fused_moe_combine_with_shared = (
            getattr(self.experts, "forward_moe_combine_with_shared", None)
            if fused_moe_combine is not None
            and self.shared_experts is not None
            and os.environ.get("MXFP4_DEEPSEEK_SHARED_ROUTED_COMBINED_X_QUANT", "0") != "0"
            else None
        )
        fused_moe_combine_unsorted = (
            getattr(self.experts, "forward_moe_combine_unsorted_scores", None)
            if fused_moe_combine is not None
            else None
        )
        top_scores_experts_sorted = None
        route_inverse_experts_sorted = None
        route_inverse_padded_experts_sorted = None
        route_full_producer_used = False
        if (
            not self.router._debug_force_load_balance
            and fused_moe_combine_unsorted is not None
            and _mxfp4_deepseek_route_scores_full_producer(self.reorderer.num_experts)
        ):
            try:
                (
                    top_scores_experts_sorted,
                    token_indices_experts_sorted,
                    num_tokens_per_expert,
                    route_positions_experts_sorted,
                    route_inverse_experts_sorted,
                    route_inverse_padded_experts_sorted,
                ) = _MXFP4MoERouteScoresFunction.apply(
                    top_scores,
                    selected_experts_indices,
                    self.reorderer.num_experts,
                    self.reorderer.top_k,
                    _mxfp4_deepseek_grouped_m_granularity(),
                )
                _lbt_validate_ep_route_counts(
                    num_tokens_per_expert,
                    int(token_indices_experts_sorted.numel()),
                )
                if (
                    _mxfp4_deepseek_allow_unsafe_tk_ep_route()
                    and _mxfp4_deepseek_validate_tk_ep_route()
                    and not _lbt_route_positions_are_permutation(
                        route_positions_experts_sorted,
                        int(token_indices_experts_sorted.numel()),
                    )
                ):
                    raise RuntimeError("TK route producer positions are unsafe under EP")
                route_full_producer_used = True
            except (AttributeError, FileNotFoundError, ImportError, RuntimeError):
                _lbt_sync_unsafe_tk_route(selected_experts_indices)
                route_full_producer_used = False
        if route_full_producer_used:
            pass
        elif self.router._debug_force_load_balance and fused_moe_combine_unsorted is not None:
            (
                token_indices_experts_sorted,
                num_tokens_per_expert,
                route_positions_experts_sorted,
            ) = self.reorderer.forward_debug_force_load_balance_no_scores(top_scores)
        elif self.router._debug_force_load_balance:
            (
                top_scores_experts_sorted,
                token_indices_experts_sorted,
                num_tokens_per_expert,
                route_positions_experts_sorted,
            ) = self.reorderer.forward_debug_force_load_balance(top_scores)
        elif fused_moe_combine_unsorted is not None:
            (
                token_indices_experts_sorted,
                num_tokens_per_expert,
                route_positions_experts_sorted,
            ) = self.reorderer.forward_with_route_positions_no_scores(selected_experts_indices)
        elif fused_moe_combine is not None:
            (
                top_scores_experts_sorted,
                token_indices_experts_sorted,
                num_tokens_per_expert,
                route_positions_experts_sorted,
            ) = self.reorderer.forward_with_route_positions(top_scores, selected_experts_indices)
        elif _mxfp4_deepseek_route_scores_full_fallback():
            try:
                (
                    top_scores_experts_sorted,
                    token_indices_experts_sorted,
                    num_tokens_per_expert,
                    route_positions_experts_sorted,
                    route_inverse_experts_sorted,
                    route_inverse_padded_experts_sorted,
                ) = _MXFP4MoERouteScoresFunction.apply(
                    top_scores,
                    selected_experts_indices,
                    self.reorderer.num_experts,
                    self.reorderer.top_k,
                    _mxfp4_deepseek_grouped_m_granularity(),
                )
                _lbt_validate_ep_route_counts(
                    num_tokens_per_expert,
                    int(token_indices_experts_sorted.numel()),
                )
                if (
                    _mxfp4_deepseek_allow_unsafe_tk_ep_route()
                    and _mxfp4_deepseek_validate_tk_ep_route()
                    and not _lbt_route_positions_are_permutation(
                        route_positions_experts_sorted,
                        int(token_indices_experts_sorted.numel()),
                    )
                ):
                    raise RuntimeError("TK route producer positions are unsafe under EP")
                route_full_producer_used = True
            except (AttributeError, FileNotFoundError, ImportError, RuntimeError):
                _lbt_sync_unsafe_tk_route(selected_experts_indices)
                (
                    top_scores_experts_sorted,
                    token_indices_experts_sorted,
                    num_tokens_per_expert,
                ) = self.reorderer(top_scores, selected_experts_indices)
                route_positions_experts_sorted = None
        else:
            if _mxfp4_deepseek_tk_route_inverse_fallback_combine():
                (
                    top_scores_experts_sorted,
                    token_indices_experts_sorted,
                    num_tokens_per_expert,
                    route_positions_experts_sorted,
                ) = self.reorderer.forward_with_route_positions(
                    top_scores,
                    selected_experts_indices,
                )
            else:
                (
                    top_scores_experts_sorted,
                    token_indices_experts_sorted,
                    num_tokens_per_expert,
                ) = self.reorderer(top_scores, selected_experts_indices)
                route_positions_experts_sorted = None

        route_metadata_maybe_unsafe = route_full_producer_used or (
            _mxfp4_deepseek_allow_unsafe_tk_ep_route()
            and route_positions_experts_sorted is not None
        )
        if _lbt_env_flag("LBT_MOE_ROUTE_METADATA_DEBUG"):
            with torch.no_grad():
                print(
                    "[lbt_moe_route_metadata]"
                    f" rank={_lbt_dist_rank()}"
                    f" world={_lbt_dist_world_size()}"
                    f" unsafe={int(_mxfp4_deepseek_allow_unsafe_tk_ep_route())}"
                    f" maybe_unsafe={int(route_metadata_maybe_unsafe)}"
                    f" full={int(route_full_producer_used)}"
                    f" route_pos={int(route_positions_experts_sorted is not None)}"
                    f" routes={int(token_indices_experts_sorted.reshape(-1).numel())}"
                    f" selected={int(selected_experts_indices.reshape(-1).numel())}"
                    f" count_dtype={num_tokens_per_expert.dtype}"
                    f" count_sum={int(num_tokens_per_expert.to(torch.int64).sum().item())}",
                    flush=True,
                )
        route_count = int(token_indices_experts_sorted.reshape(-1).numel())
        route_metadata_valid = True
        if route_metadata_maybe_unsafe and _mxfp4_deepseek_validate_tk_ep_route():
            route_metadata_valid = _lbt_ep_route_counts_match(
                num_tokens_per_expert,
                route_count,
            ) and (
                not _mxfp4_deepseek_allow_unsafe_tk_ep_route()
                or route_positions_experts_sorted is None
                or _lbt_route_positions_are_permutation(
                    route_positions_experts_sorted,
                    route_count,
                )
            )
        if route_metadata_maybe_unsafe and not route_metadata_valid:
            _lbt_sync_unsafe_tk_route(selected_experts_indices)
            (
                token_indices_experts_sorted,
                num_tokens_per_expert,
                route_positions_experts_sorted,
            ) = self.reorderer.forward_with_route_positions_no_scores_torch(
                selected_experts_indices
            )
            top_scores_experts_sorted = None
            route_inverse_experts_sorted = None
            route_inverse_padded_experts_sorted = None
            route_full_producer_used = False
            if _lbt_env_flag("LBT_MOE_ROUTE_METADATA_DEBUG"):
                with torch.no_grad():
                    print(
                        "[lbt_moe_route_metadata]"
                        f" rank={_lbt_dist_rank()}"
                        " fallback=1"
                        f" routes={int(token_indices_experts_sorted.reshape(-1).numel())}"
                        f" count_sum={int(num_tokens_per_expert.to(torch.int64).sum().item())}",
                        flush=True,
                    )

        if (
            route_metadata_maybe_unsafe
            and _mxfp4_deepseek_allow_unsafe_tk_ep_route()
        ):
            num_tokens_per_expert = num_tokens_per_expert.to(torch.int64)

        if (
            route_inverse_experts_sorted is None
            and route_positions_experts_sorted is not None
            and _mxfp4_deepseek_tk_route_inverse_fallback_combine()
        ):
            build_inverse_fn = _get_mxfp4_moe_build_route_inverse_fn()
            if build_inverse_fn is not None:
                try:
                    route_inverse_experts_sorted = build_inverse_fn(
                        _mxfp4_as_contiguous_dtype(
                            route_positions_experts_sorted.reshape(-1),
                            torch.int64,
                        )
                    )
                except (AttributeError, FileNotFoundError, ImportError):
                    route_inverse_experts_sorted = None

        if skip_router_histc:
            with torch.no_grad():
                self.tokens_per_expert.add_(num_tokens_per_expert)

        fused_kwargs = {}
        if indexed_x_rms_base is not None:
            fused_kwargs = {
                "indexed_x_rms_base": indexed_x_rms_base,
                "indexed_x_rms_weight": indexed_x_rms_weight,
                "indexed_x_rms_inv": indexed_x_rms_inv,
            }

        if fused_moe_combine_with_shared is not None:
            if top_scores_experts_sorted is None:
                top_scores_experts_sorted = _mxfp4_gather_route_scores(
                    top_scores,
                    route_positions_experts_sorted,
                )
            fused_out = fused_moe_combine_with_shared(
                x,
                self.shared_experts,
                top_scores_experts_sorted,
                token_indices_experts_sorted,
                num_tokens_per_expert,
                route_positions_experts_sorted,
                route_inverse_experts_sorted,
                route_inverse_padded_experts_sorted,
                **fused_kwargs,
            )
            if fused_out is not None:
                if forward_debug_enabled:
                    _lbt_moe_forward_debug(
                        self,
                        "fused_moe_with_shared_output",
                        output=fused_out,
                    )
                return fused_out.reshape(bs, slen, dim)

        if fused_moe_combine_unsorted is not None and not route_full_producer_used:
            fused_out = fused_moe_combine_unsorted(
                x,
                top_scores,
                token_indices_experts_sorted,
                num_tokens_per_expert,
                route_positions_experts_sorted,
                route_inverse_experts_sorted,
                route_inverse_padded_experts_sorted,
                **fused_kwargs,
            )
            if fused_out is not None:
                if self.shared_experts is not None:
                    fused_out = fused_out + self.shared_experts(x)
                if forward_debug_enabled:
                    _lbt_moe_forward_debug(
                        self,
                        "fused_moe_unsorted_output",
                        output=fused_out,
                    )
                return fused_out.reshape(bs, slen, dim)

        if fused_moe_combine is not None:
            if top_scores_experts_sorted is None:
                top_scores_experts_sorted = _mxfp4_gather_route_scores(
                    top_scores,
                    route_positions_experts_sorted,
                )
            fused_out = fused_moe_combine(
                x,
                top_scores_experts_sorted,
                token_indices_experts_sorted,
                num_tokens_per_expert,
                route_positions_experts_sorted,
                route_inverse_experts_sorted,
                route_inverse_padded_experts_sorted,
                **fused_kwargs,
            )
            if fused_out is not None:
                if self.shared_experts is not None:
                    fused_out = fused_out + self.shared_experts(x)
                if forward_debug_enabled:
                    _lbt_moe_forward_debug(
                        self,
                        "fused_moe_sorted_output",
                        output=fused_out,
                    )
                return fused_out.reshape(bs, slen, dim)

        use_indexed_fallback_combine = _mxfp4_deepseek_indexed_fallback_combine()
        if use_indexed_fallback_combine:
            # shape (bs*slen*top_k)
            token_indices_experts_sorted = token_indices_experts_sorted.reshape(-1)

            # shape (bs*slen*top_k, dim)
            routed_input = x.index_select(dim=0, index=token_indices_experts_sorted)
        else:
            # shape (bs*slen*top_k, dim)
            token_indices_experts_sorted = token_indices_experts_sorted.reshape(
                -1, 1
            ).expand(-1, dim)

            # shape (bs*slen*top_k, dim)
            routed_input = torch.gather(
                x,
                dim=0,
                index=token_indices_experts_sorted,
            )

        if self.score_before_experts:
            routed_input = (
                routed_input.to(torch.float32)
                * top_scores_experts_sorted.reshape(-1, 1)
            ).to(x.dtype)

        if _lbt_env_flag("LBT_MOE_ROUTE_METADATA_DEBUG"):
            with torch.no_grad():
                print(
                    "[lbt_moe_route_metadata]"
                    f" rank={_lbt_dist_rank()}"
                    " pre_experts=1"
                    f" routed_rows={int(routed_input.shape[0])}"
                    f" count_dtype={num_tokens_per_expert.dtype}"
                    f" count_sum={int(num_tokens_per_expert.to(torch.int64).sum().item())}"
                    f" counts={num_tokens_per_expert.to(torch.int64).tolist()}",
                    flush=True,
                )

        # shape (bs*slen*top_k, dim)
        routed_output = self.experts(routed_input, num_tokens_per_expert)

        # shared expert
        # Note: we execute the shared expert before scoring the output of the routed expert
        # to "implicitly" overlap the shared expert compute with token combine communication
        if self.shared_experts is not None:
            out = self.shared_experts(x)
        else:
            out = torch.zeros_like(x)

        if (
            use_indexed_fallback_combine
            and not self.score_before_experts
            and _mxfp4_deepseek_scored_indexed_fallback_combine()
        ):
            out = _MoEScoredIndexCombineFunction.apply(
                out,
                token_indices_experts_sorted,
                routed_output,
                top_scores_experts_sorted,
                route_inverse_experts_sorted,
                self.reorderer.top_k,
            )
        else:
            if not self.score_before_experts:
                routed_output = (
                    routed_output.to(torch.float32)
                    * top_scores_experts_sorted.reshape(-1, 1)
                ).to(x.dtype)
            if use_indexed_fallback_combine:
                out = _MoEIndexCombineFunction.apply(
                    out,
                    token_indices_experts_sorted,
                    routed_output,
                )
            else:
                out = out.scatter_add(
                    dim=0,
                    index=token_indices_experts_sorted,
                    src=routed_output,
                )
        out = out.reshape(bs, slen, dim)
        if forward_debug_enabled:
            _lbt_moe_forward_debug(
                self,
                "fallback_output",
                output=out,
            )
        return out

    def init_weights(
        self,
        init_std: float,
        buffer_device: torch.device,
    ):
        self.experts.init_weights(init_std)
        self.router.init_weights(init_std)
        if "LBT_MOE_INIT_DEBUG" in os.environ:
            _lbt_moe_init_debug(self, init_std)
        if self.shared_experts is not None:
            self.shared_experts.init_weights(init_std)
        mxfp4_ffn_norm = getattr(self, "_mxfp4_ffn_norm", None)
        if (
            _lbt_reset_moved_ffn_norm()
            and mxfp4_ffn_norm is not None
            and hasattr(mxfp4_ffn_norm, "reset_parameters")
        ):
            mxfp4_ffn_norm.reset_parameters()

        with torch.device(buffer_device):
            self.tokens_per_expert = torch.zeros(
                self.experts.num_experts, dtype=torch.float32
            )
            if self.load_balance_coeff is not None:
                self.expert_bias = torch.zeros(
                    self.experts.num_experts, dtype=torch.float32
                )
