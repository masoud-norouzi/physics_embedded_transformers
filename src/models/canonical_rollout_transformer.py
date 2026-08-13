from __future__ import annotations

import copy

import torch
from torch import nn


class CanonicalRolloutTransformer(nn.Module):
    """Multi-droplet attention model copied from the previous rollout trainer."""

    def __init__(
        self,
        input_dim=5,
        target_dim=2,
        T_history=20,
        max_droplets=64,
        d_model=128,
        n_heads=4,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        predict_branch_decision=False,
        decision_stop_gradient=False,
        condition_velocity_on_decision=False,
    ):
        super().__init__()

        if condition_velocity_on_decision and not predict_branch_decision:
            raise ValueError("condition_velocity_on_decision=True requires predict_branch_decision=True")

        self.input_dim = input_dim
        self.target_dim = target_dim
        self.T_history = T_history
        self.max_droplets = max_droplets
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.predict_branch_decision = predict_branch_decision
        self.decision_stop_gradient = decision_stop_gradient
        self.condition_velocity_on_decision = condition_velocity_on_decision

        self.droplet_mlp = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )

        self.time_embedding = nn.Embedding(T_history, d_model)
        self.slot_embedding = nn.Embedding(max_droplets, d_model)
        self.mask_embedding = nn.Embedding(2, d_model)

        encoder_layer = AttentionRecordingTransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = AttentionRecordingTransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        velocity_head_input_dim = d_model + 1 if condition_velocity_on_decision else d_model
        self.velocity_head = nn.Sequential(
            nn.Linear(velocity_head_input_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, target_dim),
        )

        self.decision_head = (
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
                nn.Linear(d_model, 1),
            )
            if predict_branch_decision
            else None
        )

    def forward(
        self,
        history_x: torch.Tensor,
        history_mask: torch.Tensor,
        return_attention: bool = False,
        return_decision: bool = False,
        attention_layer: str | int = "final",
        decision_condition_signal: torch.Tensor | None = None,
        decision_condition_gate: torch.Tensor | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor | list[torch.Tensor]]:
        B, T, M, F = history_x.shape
        assert T == self.T_history
        assert M == self.max_droplets
        assert F == self.input_dim
        assert history_mask.shape == (B, T, M)

        h = self.droplet_mlp(history_x)

        time_ids = torch.arange(T, device=history_x.device)
        slot_ids = torch.arange(M, device=history_x.device)
        time_embedding = self.time_embedding(time_ids).view(1, T, 1, self.d_model)
        slot_embedding = self.slot_embedding(slot_ids).view(1, 1, M, self.d_model)
        mask_embedding = self.mask_embedding(history_mask.long())

        h = h + time_embedding + slot_embedding + mask_embedding
        h = h.reshape(B, T * M, self.d_model)

        src_key_padding_mask = (~history_mask).reshape(B, T * M)
        if return_attention:
            h, attention_layers = self.transformer(
                h,
                src_key_padding_mask=src_key_padding_mask,
                return_attention=True,
            )
        else:
            h = self.transformer(
                h,
                src_key_padding_mask=src_key_padding_mask,
            )
            attention_layers = None

        h = h.reshape(B, T, M, self.d_model)
        h_last = h[:, -1, :, :]

        decision_logit = None
        if self.condition_velocity_on_decision or return_decision:
            if self.decision_head is None:
                raise ValueError("return_decision=True requires predict_branch_decision=True at construction")
            decision_input = h_last.detach() if self.decision_stop_gradient else h_last
            decision_logit = self.decision_head(decision_input).squeeze(-1)

        if self.condition_velocity_on_decision:
            effective_signal = torch.sigmoid(decision_logit)
            if decision_condition_signal is not None:
                effective_signal = torch.where(
                    torch.isfinite(decision_condition_signal), decision_condition_signal, effective_signal
                )
            if decision_condition_gate is not None:
                neutral = torch.full_like(effective_signal, 0.5)
                effective_signal = torch.where(decision_condition_gate, effective_signal, neutral)
            velocity_head_input = torch.cat([h_last, effective_signal.unsqueeze(-1)], dim=-1)
        else:
            velocity_head_input = h_last

        prediction = self.velocity_head(velocity_head_input)
        if not return_attention and not return_decision and not self.condition_velocity_on_decision:
            return prediction

        output: dict[str, torch.Tensor | list[torch.Tensor]] = {"prediction": prediction}
        if return_attention:
            output["attention"] = select_attention_layer(attention_layers, attention_layer)
            output["attention_layers"] = attention_layers
        if return_decision or self.condition_velocity_on_decision:
            output["decision_logit"] = decision_logit
        return output


class AttentionRecordingTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: nn.Module, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = None

    def forward(
        self,
        src: torch.Tensor,
        mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        output = src
        attention_layers = []

        for layer in self.layers:
            if return_attention:
                output, attention = layer(
                    output,
                    src_mask=mask,
                    src_key_padding_mask=src_key_padding_mask,
                    return_attention=True,
                )
                attention_layers.append(attention)
            else:
                output = layer(
                    output,
                    src_mask=mask,
                    src_key_padding_mask=src_key_padding_mask,
                )

        if self.norm is not None:
            output = self.norm(output)

        if return_attention:
            return output, attention_layers
        return output


class AttentionRecordingTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        is_causal: bool = False,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if not return_attention:
            return super().forward(
                src,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                is_causal=is_causal,
            )

        x = src
        if self.norm_first:
            attention_output, attention = self._sa_block_with_attention(
                self.norm1(x),
                src_mask,
                src_key_padding_mask,
                is_causal=is_causal,
            )
            x = x + attention_output
            x = x + self._ff_block(self.norm2(x))
        else:
            attention_output, attention = self._sa_block_with_attention(
                x,
                src_mask,
                src_key_padding_mask,
                is_causal=is_causal,
            )
            x = self.norm1(x + attention_output)
            x = self.norm2(x + self._ff_block(x))

        return x, attention

    def _sa_block_with_attention(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, attention = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
            is_causal=is_causal,
        )
        return self.dropout1(x), attention


def select_attention_layer(attention_layers: list[torch.Tensor], layer: str | int) -> torch.Tensor:
    if layer == "final":
        return attention_layers[-1]
    layer_index = int(layer)
    if layer_index < 0:
        layer_index += len(attention_layers)
    if layer_index < 0 or layer_index >= len(attention_layers):
        raise IndexError(f"Attention layer {layer} is outside 0..{len(attention_layers) - 1}.")
    return attention_layers[layer_index]
