from __future__ import annotations

import copy
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn.utils.stateless import functional_call

from hyper_har.backbone.tinierhar import TinierHAR


AdapterDict = Dict[str, Tuple[torch.Tensor, torch.Tensor]]


class TinierHARWrapper(nn.Module):
    """Differentiable TinierHAR + HyperNet LoRA composition.

    This wrapper does not mutate model weights. Instead, it builds an adapted
    parameter mapping per forward pass and evaluates the model via
    ``functional_call`` so gradients flow into adapter tensors.
    """

    def __init__(
        self,
        tinierhar: TinierHAR,
        adapters: AdapterDict | None = None,
        adapter_index: int = 0,
        lora_alpha: float = 1.0,
        lora_rank: int = 8,
        lora_scale: float | None = None,
        copy_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.model = copy.deepcopy(tinierhar) if copy_backbone else tinierhar
        self.adapters = adapters
        self.adapter_index = adapter_index
        self.lora_scale = (
            float(lora_scale)
            if lora_scale is not None
            else float(lora_alpha / max(1, lora_rank))
        )
        self.target_param_names = {
            "conv1_pointwise": "conv_blocks.0.conv.0.pointwise.weight",
            "conv_last_pointwise": (
                f"conv_blocks.{len(self.model.conv_blocks) - 1}.conv.0.pointwise.weight"
            ),
            "gru_ih_fwd": "gru.weight_ih_l0",
            "gru_ih_rev": "gru.weight_ih_l0_reverse",
            "classifier": "classifier.0.weight",
        }

    def _compute_delta(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Linear LoRA delta: B(out, r) @ A(r, in) -> (out, in)
        return (B @ A) * self.lora_scale

    def _select_pair(
        self, name: str, adapters: AdapterDict, expected_dims: tuple[int, int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if name not in adapters:
            raise KeyError(f"Missing adapter '{name}' in adapters dict.")

        A, B = adapters[name]
        a_dim, b_dim = expected_dims

        # If batched adapters are provided, pick the requested subject index.
        if A.dim() == a_dim + 1 and B.dim() == b_dim + 1:
            if A.size(0) <= self.adapter_index or B.size(0) <= self.adapter_index:
                raise IndexError(
                    f"adapter_index={self.adapter_index} out of range for '{name}'."
                )
            A = A[self.adapter_index]
            B = B[self.adapter_index]

        if A.dim() != a_dim or B.dim() != b_dim:
            raise ValueError(
                f"Unexpected shapes for '{name}': A={tuple(A.shape)}, B={tuple(B.shape)}"
            )
        return A, B

    def _build_adapted_params(self, adapters: AdapterDict) -> Dict[str, torch.Tensor]:
        params = dict(self.model.named_parameters())

        # conv1_pointwise (conv weight shape: out, in, 1, 1)
        A, B = self._select_pair("conv1_pointwise", adapters, expected_dims=(4, 4))
        delta = self._compute_delta(A.squeeze(-1).squeeze(-1), B.squeeze(-1).squeeze(-1))
        key = self.target_param_names["conv1_pointwise"]
        params[key] = params[key] + delta.unsqueeze(-1).unsqueeze(-1)

        # conv_last_pointwise
        A, B = self._select_pair("conv_last_pointwise", adapters, expected_dims=(4, 4))
        delta = self._compute_delta(A.squeeze(-1).squeeze(-1), B.squeeze(-1).squeeze(-1))
        key = self.target_param_names["conv_last_pointwise"]
        params[key] = params[key] + delta.unsqueeze(-1).unsqueeze(-1)

        # gru input-hidden weights (forward + reverse)
        A, B = self._select_pair("gru_ih_fwd", adapters, expected_dims=(2, 2))
        key = self.target_param_names["gru_ih_fwd"]
        params[key] = params[key] + self._compute_delta(A, B)

        A, B = self._select_pair("gru_ih_rev", adapters, expected_dims=(2, 2))
        key = self.target_param_names["gru_ih_rev"]
        params[key] = params[key] + self._compute_delta(A, B)

        # classifier weight
        A, B = self._select_pair("classifier", adapters, expected_dims=(2, 2))
        key = self.target_param_names["classifier"]
        params[key] = params[key] + self._compute_delta(A, B)

        return params

    def forward(
        self, x: torch.Tensor, adapters: AdapterDict | None = None
    ) -> torch.Tensor:
        active_adapters = adapters if adapters is not None else self.adapters
        if active_adapters is None:
            raise ValueError(
                "No adapters provided. Pass adapters in the constructor or forward()."
            )

        adapted_params = self._build_adapted_params(active_adapters)
        return functional_call(self.model, adapted_params, (x,))
