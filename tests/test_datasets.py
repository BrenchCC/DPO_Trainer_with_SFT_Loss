"""Tests for SFT and DPO dataset preprocessing."""

import os
import sys
import json
import tempfile
import unittest
from typing import Any, Dict, List

sys.path.append(os.getcwd())

from data.datasets import DPODataset, DPODataCollator, SupervisedDataset


class FakeTokenizer:
    """Small deterministic tokenizer implementing the required chat API."""

    eos_token_id = 99
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encode characters into stable positive token IDs.

        Args:
            text: Text to encode.
            add_special_tokens: Unused compatibility flag.

        Returns:
            Deterministic character token IDs.
        """
        del add_special_tokens
        return [10 + ord(character) % 40 for character in text]

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        tokenize: bool,
        add_generation_prompt: bool,
        return_dict: bool
    ) -> List[int]:
        """Render role-content messages into deterministic token IDs.

        Args:
            messages: Chat messages to render.
            tokenize: Must be true for dataset preprocessing.
            add_generation_prompt: Whether to append an assistant marker.
            return_dict: Must be false for a plain token list.

        Returns:
            Rendered prompt token IDs.
        """
        if not tokenize or return_dict:
            raise ValueError("FakeTokenizer only returns token lists.")

        role_tokens = {"system": 1, "user": 2, "assistant": 3}
        token_ids = []
        for message in messages:
            token_ids.append(role_tokens[message["role"]])
            token_ids.extend(self.encode(message["content"]))
        if add_generation_prompt:
            token_ids.append(4)

        return token_ids


class DatasetTestCase(unittest.TestCase):
    """Exercise dataset expansion, truncation, and collation."""

    def setUp(self) -> None:
        """Create temporary JSON file tracking for each test."""
        self.temp_paths = []
        self.tokenizer = FakeTokenizer()

    def tearDown(self) -> None:
        """Remove temporary JSON files created by a test."""
        for temp_path in self.temp_paths:
            os.remove(temp_path)

    def _write_json(self, data: List[Dict[str, Any]]) -> str:
        """Write test data to a temporary JSON file.

        Args:
            data: Records serialized to the temporary file.

        Returns:
            Temporary file path.
        """
        with tempfile.NamedTemporaryFile(
            mode = "w",
            suffix = ".json",
            encoding = "utf-8",
            delete = False
        ) as file:
            json.dump(data, file, ensure_ascii = False)
            self.temp_paths.append(file.name)

        return file.name

    def test_supervised_dataset_expands_every_assistant_turn(self) -> None:
        """Verify that each assistant message becomes a masked SFT sample."""
        data_path = self._write_json(
            [
                {
                    "messages": [
                        {"role": "user", "content": "a"},
                        {"role": "assistant", "content": "b"},
                        {"role": "user", "content": "c"},
                        {"role": "assistant", "content": "de"}
                    ]
                }
            ]
        )
        dataset = SupervisedDataset(data_path, self.tokenizer, 32)

        self.assertEqual(len(dataset), 2)
        for sample in dataset:
            first_label = next(
                index for index, label in enumerate(sample["labels"])
                if label != -100
            )
            self.assertGreater(first_label, 0)
            self.assertTrue(all(label == -100 for label in sample["labels"][:first_label]))
            self.assertEqual(sample["labels"][-1], self.tokenizer.eos_token_id)
            self.assertEqual(len(sample["input_ids"]), len(sample["labels"]))

    def test_dpo_truncation_preserves_common_prompt_and_eos(self) -> None:
        """Verify paired truncation retains a shared prompt and terminal EOS."""
        data_path = self._write_json(
            [
                {
                    "instruction": "long prompt",
                    "chosen": "abcdefghijk",
                    "rejected": "xy",
                    "history": [["old user", "old assistant"]]
                }
            ]
        )
        sample = DPODataset(data_path, self.tokenizer, 8)[0]
        chosen_start = sample["chosen_start_position"]
        rejected_start = sample["rejected_start_position"]

        self.assertEqual(chosen_start, rejected_start)
        self.assertEqual(chosen_start, 0)
        self.assertEqual(
            sample["chosen_input_ids"][:chosen_start + 1],
            sample["rejected_input_ids"][:rejected_start + 1]
        )
        self.assertEqual(sample["chosen_input_ids"][-1], self.tokenizer.eos_token_id)
        self.assertEqual(sample["rejected_input_ids"][-1], self.tokenizer.eos_token_id)
        self.assertLessEqual(len(sample["chosen_input_ids"]), 8)
        self.assertEqual(sample["labels"][0], -100)

    def test_dpo_history_accepts_message_and_pair_formats(self) -> None:
        """Verify both supported DPO history representations are encoded."""
        data_path = self._write_json(
            [
                {
                    "instruction": "question",
                    "chosen": "good",
                    "rejected": "bad",
                    "history": [{"role": "system", "content": "system"}]
                },
                {
                    "instruction": "question",
                    "chosen": "good",
                    "rejected": "bad",
                    "history": [["user", "assistant"]]
                }
            ]
        )

        self.assertEqual(len(DPODataset(data_path, self.tokenizer, 32)), 2)

    def test_dpo_collator_pads_pairs_and_labels(self) -> None:
        """Verify collator keys, masks, padding, and label padding."""
        data_path = self._write_json(
            [
                {
                    "instruction": "q1",
                    "chosen": "long",
                    "rejected": "x"
                },
                {
                    "instruction": "q2",
                    "chosen": "y",
                    "rejected": "longer"
                }
            ]
        )
        dataset = DPODataset(data_path, self.tokenizer, 32)
        batch = DPODataCollator(self.tokenizer)([dataset[0], dataset[1]])

        expected_keys = {
            "chosen_input_ids",
            "chosen_attention_mask",
            "chosen_start_position",
            "rejected_input_ids",
            "rejected_attention_mask",
            "rejected_start_position",
            "labels"
        }
        self.assertEqual(set(batch), expected_keys)
        self.assertEqual(batch["chosen_input_ids"].shape[0], 2)
        self.assertEqual(batch["rejected_input_ids"].shape[0], 2)
        self.assertTrue(
            (batch["labels"][batch["chosen_attention_mask"] == 0] == -100).all()
        )


if __name__ == "__main__":
    unittest.main()
