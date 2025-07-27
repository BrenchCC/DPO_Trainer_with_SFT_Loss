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
