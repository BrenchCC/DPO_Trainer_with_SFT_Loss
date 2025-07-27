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
