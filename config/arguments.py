"""Dataclass arguments used by the training entry point."""

from typing import Optional
from dataclasses import dataclass, field

from transformers import TrainingArguments as HFTrainingArguments


@dataclass
class ModelArguments:
    """Model path configuration."""

    model_name_or_path: str = field(
        default = "Qwen/Qwen3-0.6B",
        metadata = {"help": "Base pretrained model path or Hugging Face model ID."}
    )


@dataclass
class DataArguments:
    """Training and evaluation dataset paths."""

    train_data_path: str = field(
        default = "data/train.json",
        metadata = {"help": "Path to the training JSON file."}
    )
    eval_data_path: Optional[str] = field(
        default = None,
        metadata = {"help": "Optional path to the evaluation JSON file."}
    )


@dataclass
class TrainingArguments(HFTrainingArguments):
    """Project training arguments extending Hugging Face arguments."""

    cache_dir: Optional[str] = field(
        default = None,
        metadata = {"help": "Optional Hugging Face cache directory."}
    )
    model_max_length: int = field(
        default = 512,
        metadata = {"help": "Maximum token length after truncation."}
    )
    lora: bool = field(
        default = False,
        metadata = {
            "help": "Enable LoRA fine-tuning.",
            "aliases": ["--use_lora", "--use-lora"]
        }
    )
    qlora: bool = field(
        default = False,
        metadata = {"help": "Enable 4-bit QLoRA fine-tuning."}
    )
    dpo: bool = field(
        default = False,
        metadata = {
            "help": "Enable DPO instead of standard SFT.",
            "aliases": ["--use_dpo", "--use-dpo"]
        }
    )
    dpo_beta: float = field(
        default = 0.1,
        metadata = {"help": "DPO beta controlling policy-reference divergence."}
    )
    reference_model_path: Optional[str] = field(
        default = None,
        metadata = {"help": "Optional DPO reference model path."}
    )
    sft_loss_weight: float = field(
        default = 0.1,
        metadata = {"help": "Weight of chosen-response SFT loss during DPO."}
    )
