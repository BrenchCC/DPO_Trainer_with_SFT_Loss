"""Tests for DPO probability masking and combined losses."""

import os
import sys
import shutil
import tempfile
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.getcwd())

from data.datasets import DPODataCollator
from trainers.dpo_trainer import DPOTrainer
from config.arguments import TrainingArguments


class TinyCausalLM(nn.Module):
    """Minimal trainable causal language model used by offline tests."""

    def __init__(self, vocabulary_size: int = 16):
        """Initialize a tiny embedding and output projection.

        Args:
            vocabulary_size: Number of supported token IDs.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, 8)
        self.output = nn.Linear(8, vocabulary_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None
    ) -> SimpleNamespace:
        """Return causal language modeling logits.

        Args:
            input_ids: Batch token IDs.
            attention_mask: Batch non-padding mask.
            labels: Optional labels accepted for Trainer compatibility.

        Returns:
            Namespace containing logits.
        """
        del attention_mask, labels
        return SimpleNamespace(logits = self.output(self.embedding(input_ids)))


class FixedLogitModel(nn.Module):
    """Model returning precomputed logits for response mask tests."""

    def __init__(self, logits: torch.Tensor):
        """Store the fixed logits.

        Args:
            logits: Logits returned by forward.
        """
        super().__init__()
        self.register_buffer("fixed_logits", logits)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        """Return fixed logits for the requested batch.

        Args:
            input_ids: Batch token IDs used only for batch size.
            attention_mask: Unused attention mask.

        Returns:
            Namespace containing expanded fixed logits.
        """
        del attention_mask
        return SimpleNamespace(
            logits = self.fixed_logits.expand(input_ids.shape[0], -1, -1)
        )


class DPOTrainerTestCase(unittest.TestCase):
    """Exercise DPO math without downloading a pretrained model."""

    def _build_trainer(self, sft_loss_weight: float = 0.0) -> DPOTrainer:
        """Build a DPOTrainer with tiny policy and reference models.

        Args:
            sft_loss_weight: Weight used for the optional SFT component.

        Returns:
            Initialized DPOTrainer.
        """
        torch.manual_seed(7)
        policy_model = TinyCausalLM()
        reference_model = TinyCausalLM()
        output_dir = tempfile.mkdtemp(prefix = "dpo-trainer-test-")
        self.addCleanup(shutil.rmtree, output_dir, True)
        arguments = TrainingArguments(
            output_dir = output_dir,
            per_device_train_batch_size = 1,
            max_steps = 1,
            report_to = "none",
            use_cpu = True,
            save_strategy = "no",
            disable_tqdm = True,
            logging_strategy = "no",
            remove_unused_columns = False
        )
        train_dataset = [
            {
                "chosen_input_ids": [1, 2, 3, 4],
                "chosen_start_position": 1,
                "rejected_input_ids": [1, 2, 5, 6],
                "rejected_start_position": 1,
                "labels": [-100, -100, 3, 4]
            }
        ]
        tokenizer = SimpleNamespace(pad_token_id = 0)

        return DPOTrainer(
            model = policy_model,
            args = arguments,
            train_dataset = train_dataset,
            data_collator = DPODataCollator(tokenizer),
            reference_model = reference_model,
            dpo_beta = 0.2,
            sft_loss_weight = sft_loss_weight
        )

    def test_response_mask_includes_first_response_token(self) -> None:
        """Verify shifted response positions include the first response token."""
        trainer = self._build_trainer()
        input_ids = torch.tensor([[1, 2, 3, 4]], dtype = torch.long)
        attention_mask = torch.ones_like(input_ids)
        logits = torch.arange(4 * 8, dtype = torch.float32).view(1, 4, 8)
        model = FixedLogitModel(logits)

        actual = trainer.get_log_probabilities(
            model,
            input_ids,
            attention_mask,
            torch.tensor([1])
        )
        log_probs = F.log_softmax(logits, dim = -1)
        gathered = torch.gather(
            log_probs[:, :-1, :],
            dim = -1,
            index = input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        expected = gathered[:, 1:].sum(dim = -1)

        torch.testing.assert_close(actual, expected)

    def test_dpo_accuracy_uses_reference_adjusted_rewards(self) -> None:
        """Verify accuracy compares reference-adjusted rewards, not raw logps."""
        trainer = self._build_trainer()
        loss, accuracy, stats = trainer.compute_dpo_loss(
            torch.tensor([-5.0]),
            torch.tensor([-2.0]),
            torch.tensor([-10.0]),
            torch.tensor([-3.0])
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(accuracy.item(), 1.0)
        self.assertIn("reference_diff_mean", stats)

    def test_compute_loss_combines_dpo_and_sft(self) -> None:
        """Verify combined loss is finite, differentiable, and returns outputs."""
        trainer = self._build_trainer(sft_loss_weight = 0.25)
        batch = {
            "chosen_input_ids": torch.tensor([[1, 2, 3, 4]]),
            "chosen_attention_mask": torch.tensor([[1, 1, 1, 1]]),
            "chosen_start_position": torch.tensor([1]),
            "rejected_input_ids": torch.tensor([[1, 2, 5, 6]]),
            "rejected_attention_mask": torch.tensor([[1, 1, 1, 1]]),
            "rejected_start_position": torch.tensor([1]),
            "labels": torch.tensor([[-100, -100, 3, 4]])
        }

        loss, outputs = trainer.compute_loss(
            trainer.model,
            batch,
            return_outputs = True
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(loss.requires_grad)
        self.assertEqual(set(outputs), {"chosen_logps", "rejected_logps"})

    def test_trainer_runs_one_offline_training_step(self) -> None:
        """Verify the Hugging Face training loop completes one synthetic step."""
        trainer = self._build_trainer()
        result = trainer.train()

        self.assertEqual(result.global_step, 1)
        self.assertTrue(torch.isfinite(torch.tensor(result.training_loss)))


if __name__ == "__main__":
    unittest.main()
