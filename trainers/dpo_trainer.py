"""Custom Hugging Face Trainer for DPO with optional SFT loss."""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer


class DPOTrainer(Trainer):
    """Compute DPO loss against a frozen reference model."""

    def __init__(
        self,
        reference_model = None,
        dpo_beta: float = 0.1,
        sft_loss_weight: float = 0.0,
        **kwargs
    ):
        """Initialize the DPO trainer.

        Args:
            reference_model: Frozen model used for DPO reference log probabilities.
            dpo_beta: Scale applied to the DPO preference margin.
            sft_loss_weight: Weight applied to chosen-response SFT loss.
            **kwargs: Arguments forwarded to transformers.Trainer.
        """
        if reference_model is None:
            raise ValueError("reference_model is required for DPO training.")
        if dpo_beta <= 0:
            raise ValueError("dpo_beta must be greater than zero.")
        if sft_loss_weight < 0:
            raise ValueError("sft_loss_weight cannot be negative.")

        super().__init__(**kwargs)
        self.reference_model = reference_model
        self.dpo_beta = dpo_beta
        self.sft_loss_weight = sft_loss_weight

        self.reference_model.to(self.args.device)
        self.reference_model.eval()
        for parameter in self.reference_model.parameters():
            parameter.requires_grad = False

    def get_log_probabilities(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_start_positions: torch.Tensor
    ) -> torch.Tensor:
        """Calculate summed response-token log probabilities.

        Args:
            model: Policy or reference causal language model.
            input_ids: Padded token IDs with shape batch by sequence.
            attention_mask: Mask selecting non-padding tokens.
            response_start_positions: First response position in shifted logits.

        Returns:
            Summed response log probability for every batch item.
        """
        outputs = model(
            input_ids = input_ids,
            attention_mask = attention_mask
        )

        return self._get_log_probabilities_from_logits(
            outputs.logits,
            input_ids,
            attention_mask,
            response_start_positions
        )

    def _get_log_probabilities_from_logits(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_start_positions: torch.Tensor
    ) -> torch.Tensor:
        """Calculate response log probabilities from existing model logits.

        Args:
            logits: Causal language model logits.
            input_ids: Padded token IDs with shape batch by sequence.
            attention_mask: Mask selecting non-padding tokens.
            response_start_positions: First response position in shifted logits.

        Returns:
            Summed response log probability for every batch item.
        """
        log_probabilities = F.log_softmax(logits, dim = -1)

        shifted_log_probabilities = log_probabilities[..., :-1, :].contiguous()
        shifted_labels = input_ids[..., 1:].contiguous()
        shifted_attention = attention_mask[..., 1:].contiguous()
        gathered_log_probabilities = torch.gather(
            shifted_log_probabilities,
            dim = -1,
            index = shifted_labels.unsqueeze(-1)
        ).squeeze(-1)

        sequence_length = gathered_log_probabilities.shape[1]
        response_start_positions = torch.as_tensor(
            response_start_positions,
            device = gathered_log_probabilities.device,
            dtype = torch.long
        )
        if response_start_positions.ndim != 1:
            raise ValueError("response_start_positions must be one-dimensional.")
        if response_start_positions.shape[0] != input_ids.shape[0]:
            raise ValueError("One response start position is required per sequence.")
        if torch.any(response_start_positions < 0):
            raise ValueError("Response start positions cannot be negative.")
        if torch.any(response_start_positions >= sequence_length):
            raise ValueError("Response start positions must precede the final token.")

        token_positions = torch.arange(
            sequence_length,
            device = gathered_log_probabilities.device
        )
        response_mask = token_positions.unsqueeze(0) >= response_start_positions.unsqueeze(1)
        final_mask = shifted_attention.bool() & response_mask

        return (gathered_log_probabilities * final_mask).sum(dim = -1)

    def compute_dpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """Calculate DPO loss and detached training statistics.

        Args:
            policy_chosen_logps: Policy log probabilities for chosen responses.
            policy_rejected_logps: Policy log probabilities for rejected responses.
            reference_chosen_logps: Reference log probabilities for chosen responses.
            reference_rejected_logps: Reference log probabilities for rejected responses.

        Returns:
            DPO loss, reward accuracy, and scalar statistics.
        """
        policy_chosen_rewards = policy_chosen_logps - reference_chosen_logps
        policy_rejected_rewards = policy_rejected_logps - reference_rejected_logps
        logits = self.dpo_beta * (
            policy_chosen_rewards - policy_rejected_rewards
        )
        loss = -F.logsigmoid(logits).mean()
        accuracy = (
            policy_chosen_rewards > policy_rejected_rewards
        ).float().mean()

        with torch.no_grad():
            policy_difference = policy_chosen_logps - policy_rejected_logps
            reference_difference = (
                reference_chosen_logps - reference_rejected_logps
            )

        stats = {
            "policy_diff_mean": policy_difference.mean().item(),
            "reference_diff_mean": reference_difference.mean().item(),
            "kl_chosen_mean": policy_chosen_rewards.mean().item(),
            "kl_rejected_mean": policy_rejected_rewards.mean().item(),
            "logits_mean": logits.mean().item(),
            "policy_chosen_mean": policy_chosen_logps.mean().item(),
            "policy_rejected_mean": policy_rejected_logps.mean().item(),
            "reference_chosen_mean": reference_chosen_logps.mean().item(),
            "reference_rejected_mean": reference_rejected_logps.mean().item()
        }

        return loss, accuracy, stats

    def compute_sft_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Calculate response-only causal language modeling loss.

        Args:
            logits: Model logits with shape batch by sequence by vocabulary.
            labels: Token labels with prompt and padding positions set to -100.

        Returns:
            Cross-entropy SFT loss.
        """
        shifted_logits = logits[..., :-1, :].contiguous()
        shifted_labels = labels[..., 1:].contiguous()
        loss_function = nn.CrossEntropyLoss(ignore_index = -100)

        return loss_function(
            shifted_logits.view(-1, shifted_logits.size(-1)),
            shifted_labels.view(-1)
        )

    def compute_loss(
        self,
        model,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None
    ):
        """Calculate DPO loss and optional chosen-response SFT loss.

        Args:
            model: Policy model being optimized.
            inputs: Batch produced by DPODataCollator.
            return_outputs: Whether to return prediction outputs with the loss.
            num_items_in_batch: Optional Trainer token count, unused by DPO.

        Returns:
            Loss tensor or a loss-output tuple.
        """
        del num_items_in_batch

        policy_chosen_outputs = model(
            input_ids = inputs["chosen_input_ids"],
            attention_mask = inputs["chosen_attention_mask"]
        )
        policy_chosen_logps = self._get_log_probabilities_from_logits(
            policy_chosen_outputs.logits,
            inputs["chosen_input_ids"],
            inputs["chosen_attention_mask"],
            inputs["chosen_start_position"]
        )
        policy_rejected_logps = self.get_log_probabilities(
            model,
            inputs["rejected_input_ids"],
            inputs["rejected_attention_mask"],
            inputs["rejected_start_position"]
        )

        with torch.no_grad():
            reference_chosen_logps = self.get_log_probabilities(
                self.reference_model,
                inputs["chosen_input_ids"],
                inputs["chosen_attention_mask"],
                inputs["chosen_start_position"]
            )
            reference_rejected_logps = self.get_log_probabilities(
                self.reference_model,
                inputs["rejected_input_ids"],
                inputs["rejected_attention_mask"],
                inputs["rejected_start_position"]
            )

        dpo_loss, accuracy, stats = self.compute_dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps
        )
        total_loss = dpo_loss
        log_values = {
            "dpo_loss": dpo_loss.item(),
            "dpo_accuracy": accuracy.item(),
            **stats
        }

        if self.sft_loss_weight > 0:
            sft_loss = self.compute_sft_loss(
                policy_chosen_outputs.logits,
                inputs["labels"]
            )
            total_loss = total_loss + self.sft_loss_weight * sft_loss
            log_values.update(
                {
                    "sft_loss": sft_loss.item(),
                    "sft_loss_weight": self.sft_loss_weight
                }
            )

        log_values["total_loss"] = total_loss.item()
        self.log(log_values)

        prediction_outputs = {
            "chosen_logps": policy_chosen_logps,
            "rejected_logps": policy_rejected_logps
        }

        return (total_loss, prediction_outputs) if return_outputs else total_loss
