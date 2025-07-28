import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer
from typing import List, Dict, Any, Optional

class DPOTrainer(Trainer):
    """
    Custom DPO Trainer, Support DPO Loss and optional SFT loss calculation
    """

    def __init__(
        self,
        reference_model = None,
        dpo_meta: float = 0.1,
        sft_loss_weight: float = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.reference_model = reference_model
        self.dpo_meta = dpo_meta
        self.sft_loss_weight = sft_loss_weight
        self.sft = sft_loss_weight > 0

        if self.reference_model is not None:
            self.reference_model.to(self.args.device)
            self.reference_model.eval()
            # ensure reference model don't need gradient caculation
            for param in self.reference_model.parameters():
                param.requires_grad = False

    def get_log_probabilities(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_start_positions: List[int],
    ) -> torch.Tensor:
        """
        Calculate the log probabilities of sequence, especially for DPO training.
        """
        outputs = model(
            input_ids = input_ids,
            attention_mask = attention_mask,
        )
        logits = outputs.logits

        # calculate every token log probabilities
        log_probs = F.log_softmax(logits, dim = -1)

        # gain true token log probs
        shift_log_probs = log_probs[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_attention = attention_mask[..., 1:].contiguous()

        # collect every position log probs
        gathered_log_probs = torch.gather(
            shift_log_probs,
            dim = -1,
            index = shift_labels.unsqueeze(-1),
        ).squeeze(-1)

        # create response mask, only pay attention to the part of response text
        batch_size, seq_len = gathered_log_probs.shape
        response_mask = torch.zeros_like(gathered_log_probs, dtype = torch.bool)

        for i, start_pos in enumerate(response_start_positions):
            if start_pos < seq_len:
                response_mask[i, start_pos:] = True

        # combine with attention mask and response mask
        final_mask = shift_attention.bool() & response_mask
        masked_log_probs = gathered_log_probs * final_mask.float()

        # calculate total log probs
        sequence_log_probs = masked_log_probs.sum(dim = -1)

        return sequence_log_probs

    def compute_dpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor
    ) -> tuple:
        """
        Calculating DPO loss function
        """
        # Calculate the log probability ratio relative to the reference model
        policy_ratio_chosen = policy_chosen_logps - reference_chosen_logps
        policy_ratio_rejected = policy_rejected_logps - reference_rejected_logps

        # DPO loss: The probability of encouraging chosen replies is higher than rejected replies
        logits = self.dpo_beta * (policy_ratio_chosen - policy_ratio_rejected)
        loss = -F.logsigmoid(logits).mean()

        # calculate accuracy
        accuracy = (policy_chosen_logps > policy_rejected_logps).float().mean()

        # statistic information
        with torch.no_grad():
            policy_diff = policy_chosen_logps - policy_rejected_logps
            reference_diff = reference_chosen_logps - reference_rejected_logps
            kl_chosen = policy_chosen_logps - reference_chosen_logps
            kl_rejected = policy_rejected_logps - reference_rejected_logps

        stats = {
            'policy_diff_mean': policy_diff.mean().item(),
            'reference_diff_mean': reference_diff.mean().item(),
            'kl_chosen_mean': kl_chosen.mean().item(),
            'kl_rejected_mean': kl_rejected.mean().item(),
            'logits_mean': logits.mean().item(),
            'policy_chosen_mean': policy_chosen_logps.mean().item(),
            'policy_rejected_mean': policy_rejected_logps.mean().item(),
            'reference_chosen_mean': reference_chosen_logps.mean().item(),
            'reference_rejected_mean': reference_rejected_logps.mean().item(),
        }

        return loss, accuracy, stats