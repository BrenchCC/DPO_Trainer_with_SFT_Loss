from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments as HFTrainingArguments

@dataclass
class ModelArguments:
    """
    模型相关参数配置
    """
    model_name_or_path: Optional[str] = field(
        default = "Qwen/Qwen3-0.6B",
        metadata = {"HELP": "Base Pre-Train Model Path or Name"}
    )

@dataclass
class DataArguments:
    """
    数据集相关参数配置
    """
    train_data_path: str = field(
        default = "data/train.json",
        metadata = {"HELP": "Train Data Path"}
    )
    eval_data_path: str = field(
        default = "data/eval.json",
        metadata = {"HELP": "Eval Data Path"}
    )