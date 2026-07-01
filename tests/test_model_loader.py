"""Offline tests for model loader configuration and compatibility."""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn as nn

sys.path.append(os.getcwd())

from models.model_loader import load_model_and_tokenizer, load_model_and_toknizer


class DummyTokenizer:
    """Tokenizer stub exposing EOS and mutable padding configuration."""

    eos_token = "<eos>"
    eos_token_id = 2

    def __init__(self):
        """Initialize without a padding token."""
        self._pad_token = None
        self.pad_token_id = None
        self.padding_side = "left"
        self.model_max_length = 0

    @property
    def pad_token(self):
        """Return the configured padding token."""
        return self._pad_token

    @pad_token.setter
    def pad_token(self, value: str) -> None:
        """Set padding token and mirror its token ID.

        Args:
            value: Padding token string.
        """
        self._pad_token = value
        self.pad_token_id = self.eos_token_id


class DummyModel(nn.Module):
    """Minimal model returned by patched from_pretrained calls."""

    def __init__(self):
        """Initialize a dummy parameter and model config."""
        super().__init__()
        self.projection = nn.Linear(1, 1)
        self.config = SimpleNamespace(architectures = ["DummyForCausalLM"])


class ModelLoaderTestCase(unittest.TestCase):
    """Verify model loader arguments without network or model weights."""

    @patch("models.model_loader.AutoModelForCausalLM")
    @patch("models.model_loader.AutoTokenizer")
    def test_load_model_configures_tokenizer_and_dtype(self, tokenizer_loader: Mock, model_loader: Mock) -> None:
        """Verify corrected loader arguments and tokenizer configuration.

        Args:
            tokenizer_loader: Patched tokenizer factory.
            model_loader: Patched model factory.
        """
        tokenizer = DummyTokenizer()
        model = DummyModel()
        tokenizer_loader.from_pretrained.return_value = tokenizer
        model_loader.from_pretrained.return_value = model

        loaded_model, loaded_tokenizer = load_model_and_tokenizer(
            "dummy-model",
            max_length = 128
        )

        self.assertIs(loaded_model, model)
        self.assertIs(loaded_tokenizer, tokenizer)
        self.assertEqual(tokenizer.pad_token_id, tokenizer.eos_token_id)
        self.assertEqual(tokenizer.padding_side, "right")
        self.assertEqual(tokenizer.model_max_length, 128)
        tokenizer_loader.from_pretrained.assert_called_once_with(
            "dummy-model",
            cache_dir = None,
            trust_remote_code = True
        )
        model_kwargs = model_loader.from_pretrained.call_args.kwargs
        self.assertEqual(model_kwargs["dtype"], torch.float32)
        self.assertTrue(model_kwargs["trust_remote_code"])

    @patch("models.model_loader.AutoTokenizer")
    @patch("models.model_loader.torch.cuda.is_available", return_value = False)
    def test_qlora_fails_before_loading_tokenizer(self, cuda_available: Mock, tokenizer_loader: Mock) -> None:
        """Verify unsupported QLoRA fails before network access.

        Args:
            cuda_available: Patched CUDA availability check.
            tokenizer_loader: Patched tokenizer factory.
        """
        del cuda_available
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            load_model_and_tokenizer("dummy-model", lora = True, qlora = True)

        tokenizer_loader.from_pretrained.assert_not_called()

    @patch("models.model_loader.warnings.warn")
    @patch("models.model_loader.load_model_and_tokenizer")
    def test_deprecated_loader_alias_delegates(self, corrected_loader: Mock, warning: Mock) -> None:
        """Verify the old misspelling warns and calls the corrected loader.

        Args:
            corrected_loader: Patched corrected loader function.
            warning: Patched deprecation warning emitter.
        """
        corrected_loader.return_value = ("model", "tokenizer")
        result = load_model_and_toknizer("dummy-model")

        self.assertEqual(result, ("model", "tokenizer"))
        corrected_loader.assert_called_once()
        warning.assert_called_once()


if __name__ == "__main__":
    unittest.main()
