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
_MXFP4_MOE_SCATTER_SCORES_FN = None
_MXFP4_MOE_REORDER_IMPORT_ATTEMPTED = False


def _get_mxfp4_moe_reorder_fn():
    global _MXFP4_MOE_REORDER_FN
    _ensure_mxfp4_moe_route_imports()
    return _MXFP4_MOE_REORDER_FN


def _get_mxfp4_moe_reorder_scores_full_fn():
    _ensure_mxfp4_moe_route_imports()
    return _MXFP4_MOE_REORDER_SCORES_FULL_FN


def _get_mxfp4_moe_scatter_scores_fn():
    _ensure_mxfp4_moe_route_imports()
    return _MXFP4_MOE_SCATTER_SCORES_FN


def _ensure_mxfp4_moe_route_imports():
    global _MXFP4_MOE_REORDER_FN, _MXFP4_MOE_REORDER_SCORES_FULL_FN, _MXFP4_MOE_SCATTER_SCORES_FN, _MXFP4_MOE_REORDER_IMPORT_ATTEMPTED
    if _MXFP4_MOE_REORDER_IMPORT_ATTEMPTED:
        return
    _MXFP4_MOE_REORDER_IMPORT_ATTEMPTED = True
    try:
        from low_bits_training.quantization.mxfp4_backend import (
            mxfp4_moe_reorder_indices,
            mxfp4_moe_reorder_scores_full,
            mxfp4_moe_scatter_scores,
        )
    except (ImportError, AttributeError):
        _MXFP4_MOE_REORDER_FN = None
        _MXFP4_MOE_REORDER_SCORES_FULL_FN = None
        _MXFP4_MOE_SCATTER_SCORES_FN = None
    else:
        _MXFP4_MOE_REORDER_FN = mxfp4_moe_reorder_indices
        _MXFP4_MOE_REORDER_SCORES_FULL_FN = mxfp4_moe_reorder_scores_full
        _MXFP4_MOE_SCATTER_SCORES_FN = mxfp4_moe_scatter_scores


def _mxfp4_deepseek_grouped_m_granularity() -> int:
    raw = os.environ.get("MXFP4_DEEPSEEK_GROUPED_M_GRANULARITY", "256")
    try:
        value = int(raw)
    except ValueError:
        value = 256
    return value if value in (256, 512, 1024) else 256


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
        flat_scores = top_scores.reshape(-1).to(dtype=torch.float32).contiguous()
        selected = selected_experts_indices.to(dtype=torch.int64).contiguous()
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
            grad_flat[route_positions] = grad_sorted_scores.to(dtype=torch.float32).contiguous()
        else:
            grad_flat = scatter_fn(
                grad_sorted_scores.to(dtype=torch.float32).contiguous(),
                route_positions,
                ctx.num_scores,
            )
        return grad_flat.reshape(ctx.orig_shape), None, None, None, None


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
            _, selected_experts_indices = torch.topk(
                scores + expert_bias, k=self.top_k, dim=1
            )
            top_scores = scores.gather(dim=1, index=selected_experts_indices)
        else:
            top_scores, selected_experts_indices = torch.topk(
                scores, k=self.top_k, dim=1
            )

        if self.route_norm:
            denominator = top_scores.sum(dim=-1, keepdim=True) + 1e-20
            top_scores = top_scores / denominator
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
        nn.init.trunc_normal_(self.gate.weight, mean=0.0, std=init_std)


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
        token_indices_experts_sorted = torch.argsort(
            selected_experts_indices.view(-1), stable=True
        )

        top_scores_experts_sorted = top_scores.view(-1)[token_indices_experts_sorted]
        token_indices_experts_sorted = token_indices_experts_sorted // self.top_k

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
            and selected_experts_indices.is_cuda
            and selected_experts_indices.dtype == torch.int64
        ):
            reorder_fn = _get_mxfp4_moe_reorder_fn()
            if reorder_fn is not None:
                try:
                    return reorder_fn(
                        selected_experts_indices.contiguous(),
                        self.num_experts,
                        self.top_k,
                    )
                except (AttributeError, FileNotFoundError, ImportError, RuntimeError):
                    pass

        num_tokens_per_expert = torch.histc(
            selected_experts_indices.view(-1),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        )

        route_positions_experts_sorted = torch.argsort(
            selected_experts_indices.view(-1), stable=True
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
        top_scores_experts_sorted = top_scores.view(-1)[route_positions_experts_sorted]

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

        top_scores_experts_sorted = top_scores.reshape(-1)[route_positions_experts_sorted]

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

        fused_moe_combine = (
            getattr(self.experts, "forward_moe_combine", None)
            if not self.score_before_experts
            else None
        )
        skip_router_histc = (
            fused_moe_combine is not None
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
            and os.environ.get("MXFP4_DEEPSEEK_ROUTE_SCORES_FULL_PRODUCER", "0") != "0"
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
                route_full_producer_used = True
            except (AttributeError, FileNotFoundError, ImportError, RuntimeError):
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
        else:
            (
                top_scores_experts_sorted,
                token_indices_experts_sorted,
                num_tokens_per_expert,
            ) = self.reorderer(top_scores, selected_experts_indices)
            route_positions_experts_sorted = None

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
                top_scores_experts_sorted = top_scores.reshape(-1)[route_positions_experts_sorted]
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
                return fused_out.reshape(bs, slen, dim)

        if fused_moe_combine is not None:
            if top_scores_experts_sorted is None:
                top_scores_experts_sorted = top_scores.reshape(-1)[route_positions_experts_sorted]
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
                return fused_out.reshape(bs, slen, dim)

        # shape (bs*slen*top_k, dim)
        token_indices_experts_sorted = token_indices_experts_sorted.reshape(
            -1, 1
        ).expand(-1, dim)

        # shape (bs*slen*top_k, dim)
        routed_input = torch.gather(x, dim=0, index=token_indices_experts_sorted)

        if self.score_before_experts:
            routed_input = (
                routed_input.to(torch.float32)
                * top_scores_experts_sorted.reshape(-1, 1)
            ).to(x.dtype)

        # shape (bs*slen*top_k, dim)
        routed_output = self.experts(routed_input, num_tokens_per_expert)

        # shared expert
        # Note: we execute the shared expert before scoring the output of the routed expert
        # to "implicitly" overlap the shared expert compute with token combine communication
        if self.shared_experts is not None:
            out = self.shared_experts(x)
        else:
            out = torch.zeros_like(x)

        if not self.score_before_experts:
            routed_output = (
                routed_output.to(torch.float32)
                * top_scores_experts_sorted.reshape(-1, 1)
            ).to(x.dtype)

        out = out.scatter_add(
            dim=0, index=token_indices_experts_sorted, src=routed_output
        )
        out = out.reshape(bs, slen, dim)
        return out

    def init_weights(
        self,
        init_std: float,
        buffer_device: torch.device,
    ):
        self.experts.init_weights(init_std)
        self.router.init_weights(init_std)
        if self.shared_experts is not None:
            self.shared_experts.init_weights(init_std)

        with torch.device(buffer_device):
            self.tokens_per_expert = torch.zeros(
                self.experts.num_experts, dtype=torch.float32
            )
            if self.load_balance_coeff is not None:
                self.expert_bias = torch.zeros(
                    self.experts.num_experts, dtype=torch.float32
                )
