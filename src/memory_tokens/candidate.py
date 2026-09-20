import torch
from torch import Tensor, nn

BYTE_SCALE = 0.125
CONDITION_COUNT = 256
CONTEXT_COUNT = 8
AUXILIARY_LOSS_WEIGHT = 0.1


class CandidateMemory(nn.Module):
    """Starting implementation for candidate changes.

    Replace the write and read mechanisms. Keep their signatures unchanged.
    The same module must support every slot_count used by the protocol.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_slots: int = 128,
        memory_width: int = 24,
    ) -> None:
        super().__init__()
        del max_slots
        self.d_model = d_model
        self.memory_width = memory_width
        self.metadata_width = 2
        if memory_width <= self.metadata_width:
            raise ValueError("memory width must leave room for the action payload")
        self.action_width = d_model // 4
        left_context_width = (
            d_model - 2 * self.action_width - 2 * CONTEXT_COUNT - 1
        ) // 2
        self.action_start = (
            self.action_width + 2 * CONTEXT_COUNT + left_context_width + 1
        )
        action_payload_width = memory_width - self.metadata_width
        self.write_projection = nn.Linear(self.action_width, action_payload_width)
        self.action_decoder = nn.Linear(action_payload_width, self.action_width)
        self.reader = nn.MultiheadAttention(
            d_model,
            n_heads,
            kdim=memory_width,
            vdim=memory_width,
            batch_first=True,
            dropout=0.0,
        )
        self.read_norm = nn.LayerNorm(d_model)

    def _select_regions(
        self, policy_states: Tensor, policy_mask: Tensor, slot_count: int
    ) -> tuple[Tensor, Tensor]:
        """Return selected residual regions and their padding mask."""
        batch, n_policies, width = policy_states.shape
        if width != self.d_model:
            raise ValueError("policy-state width changed")

        key_width = self.d_model // 4
        scope_start = key_width
        estimate_start = scope_start + CONTEXT_COUNT
        # The fixed encoder puts condition identity in the key lane and the
        # original context estimates immediately after the scope lane.
        condition_codes = policy_states[..., 0]
        scope_bits = policy_states[..., scope_start : scope_start + CONTEXT_COUNT].gt(
            0.5
        )
        used_contexts = torch.zeros(
            batch,
            CONDITION_COUNT,
            CONTEXT_COUNT,
            dtype=torch.bool,
            device=policy_states.device,
        )
        active = torch.zeros(
            batch,
            n_policies,
            CONTEXT_COUNT,
            dtype=torch.bool,
            device=policy_states.device,
        )
        # Walk backwards so each policy keeps only contexts not claimed later.
        for policy_index in range(n_policies - 1, -1, -1):
            condition_index = (
                (
                    torch.round(condition_codes[:, policy_index] / BYTE_SCALE)
                    + CONDITION_COUNT // 2
                )
                .long()
                .clamp(0, CONDITION_COUNT - 1)
            )
            condition_used = used_contexts.gather(
                1,
                condition_index[:, None, None].expand(-1, 1, CONTEXT_COUNT),
            ).squeeze(1)
            available = (
                policy_mask[:, policy_index, None]
                & scope_bits[:, policy_index]
                & condition_used.logical_not()
            )
            active[:, policy_index] = available
            used_contexts.scatter_(
                1,
                condition_index[:, None, None].expand(-1, 1, CONTEXT_COUNT),
                (condition_used | available).unsqueeze(1),
            )

        # One candidate is one condition/action plus its compiled context region.
        region_states = policy_states.clone()
        region_states[..., scope_start : scope_start + CONTEXT_COUNT] = active.to(
            policy_states.dtype
        )
        # Rank regions by the traffic that remains after applying overrides.
        scores = (
            policy_states[..., estimate_start : estimate_start + CONTEXT_COUNT] * active
        ).sum(dim=-1)
        scores = scores.masked_fill(active.any(dim=-1).logical_not(), float("-inf"))
        flat_scores = scores
        if slot_count > n_policies:
            padding = slot_count - n_policies
            flat_states = torch.cat(
                (region_states, region_states.new_zeros(batch, padding, width)), dim=1
            )
            flat_scores = torch.cat(
                (
                    flat_scores,
                    flat_scores.new_full((batch, padding), float("-inf")),
                ),
                dim=1,
            )
        else:
            flat_states = region_states
        top_scores, top_indices = flat_scores.topk(slot_count, dim=1)
        del top_scores
        selected_scores = flat_scores.gather(1, top_indices)
        selected = flat_states.gather(
            1, top_indices.unsqueeze(-1).expand(-1, -1, width)
        )
        invalid = torch.isinf(selected_scores)
        selected = selected.masked_fill(invalid.unsqueeze(-1), 0.0)
        return selected, invalid

    def write(
        self, policy_states: Tensor, policy_mask: Tensor, slot_count: int
    ) -> Tensor:
        """Compile the ordered policies into the highest-value rule entries."""
        selected, invalid = self._select_regions(policy_states, policy_mask, slot_count)
        key_width = self.d_model // 4
        scope_start = key_width

        # Byte 0 is already the exact signed condition code at the boundary.
        condition_payload = selected[..., :1]

        # Byte 1 stores all eight residual-scope bits. Mapping [0, 255] to
        # [-128, 127] makes every mask exactly representable by a signed byte.
        scope_weights = 1 << torch.arange(
            CONTEXT_COUNT, device=policy_states.device, dtype=torch.long
        )
        selected_scope = selected[..., scope_start : scope_start + CONTEXT_COUNT].gt(
            0.5
        )
        scope_masks = (selected_scope.long() * scope_weights).sum(dim=-1)
        scope_payload = (
            scope_masks.to(policy_states.dtype) - CONDITION_COUNT // 2
        ).unsqueeze(-1) * BYTE_SCALE

        # The remaining bytes compress the fixed encoder's action lane only.
        action_states = selected[
            ..., self.action_start : self.action_start + self.action_width
        ]
        action_payload = self.write_projection(action_states)
        action_payload = action_payload.masked_fill(invalid.unsqueeze(-1), 0.0)
        packed = torch.cat((condition_payload, scope_payload, action_payload), dim=-1)
        # The caller performs the required signed-int8 quantization afterward.
        return packed.float()

    def auxiliary_loss(
        self, policy_states: Tensor, policy_mask: Tensor, memory: Tensor
    ) -> Tensor:
        """Reconstruct selected action states from the quantized action bytes."""
        if not self.training:
            return policy_states.new_zeros(())
        selected, invalid = self._select_regions(
            policy_states, policy_mask, memory.size(1)
        )
        target = selected[
            ..., self.action_start : self.action_start + self.action_width
        ].detach()
        reconstructed = self.action_decoder(memory[..., self.metadata_width :])
        per_slot_mse = (reconstructed - target).square().mean(dim=-1)
        valid = invalid.logical_not()
        mean_mse = (per_slot_mse * valid).sum() / valid.sum().clamp_min(1)
        return mean_mse * AUXILIARY_LOSS_WEIGHT

    def read(self, request_states: Tensor, memory: Tensor) -> Tensor:
        key_width = self.d_model // 4
        request_context = request_states[
            ..., key_width : key_width + CONTEXT_COUNT
        ].argmax(dim=-1)
        request_condition = request_states[..., 0]

        stored_condition = memory[..., 0]
        condition_matches = request_condition.unsqueeze(-1).eq(
            stored_condition.unsqueeze(1)
        )
        scope_masks = (
            torch.round(memory[..., 1] / BYTE_SCALE).long() + CONDITION_COUNT // 2
        ).clamp(0, CONDITION_COUNT - 1)
        requested_bits = 1 << request_context.unsqueeze(-1)
        scope_matches = scope_masks.unsqueeze(1).bitwise_and(requested_bits).ne(0)
        compatible = condition_matches & scope_matches

        # Avoid an all-masked attention row when the budget omitted a context.
        no_compatible_slot = compatible.any(dim=-1, keepdim=True).logical_not()
        attention_mask = (~compatible & ~no_compatible_slot).repeat_interleave(
            self.reader.num_heads, dim=0
        )
        attended, _ = self.reader(
            request_states,
            memory,
            memory,
            attn_mask=attention_mask,
            need_weights=False,
        )
        attended = attended.masked_fill(no_compatible_slot, 0.0)
        return self.read_norm(request_states + attended)
