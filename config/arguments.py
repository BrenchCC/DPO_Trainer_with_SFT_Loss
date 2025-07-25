from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments as HFTrainingArguments

@dataclass
class ModelArguments:
    """
    model arguments configs
    """
    model_name_or_path: Optional[str] = field(
        default = "Qwen/Qwen3-0.6B",
        metadata = {"HELP": "Base Pre-Train Model Path or Name"}
    )

@dataclass
class DataArguments:
    """
    data arguments configs
    """
    train_data_path: str = field(
        default = "data/train.json",
        metadata = {"HELP": "Train Data Path"}
    )
    eval_data_path: str = field(
        default = "data/eval.json",
        metadata = {"HELP": "Eval Data Path"}
    )

@dataclass
class TrainingArguments(HFTrainingArguments):
    """
    training arguments configs，FROM HuggingFace TrainingArguments
    """
    cache_dir: Optional[str] = field(default=None, metadata={"HELP": "Cache Dir"})
    optimizer: str = field(default = "adamw_torch")
    model_max_length: int = field(
        default = 512,
        metadata={"HELP": "Maximum Length of Sequence To be Padded"}
    )
    lora: bool = field(default = False, metadata={"HELP": "LoRa Enable"})
    qlora: bool = field(default = False, metadata={"HELP": "QLoRa Enable"})

    # DPO related arguments
    dpo: bool = field(default = False, metadata={"HELP": "Use DPO"})
    dpo_beta: float = field(
        default = 0.1,
        metadata={"HELP": "DPO Loss Function Beta Argument, controls the degree to which the strategy model loses the reference model"}
    )
    reference_model_path: Optional[str] = field(
        default = None,
        metadata={"HELP": "Reference Model Path, if None then using main model"}
    )

    # SFT Loss Weight Arguments
    sft_loss_weight: float = field(
        default = 0.1,
        metadata={"HELP": "SFT Loss Weight, combine with DPO Training"}
    )
if __name__ == "__main__":
    print(DataArguments.train_data_path)
    print(DataArguments.eval_data_path)