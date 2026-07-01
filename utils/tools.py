"""Validation, configuration, and training summary helpers."""

import os
import json
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def create_output_dir(output_dir: str) -> None:
    """Create the training output directory.

    Args:
        output_dir: Directory created when missing.
    """
    os.makedirs(output_dir, exist_ok = True)


def save_training_config(config: Dict[str, Any], output_dir: str) -> None:
    """Save the resolved training configuration as JSON.

    Args:
        config: Serializable training configuration.
        output_dir: Directory receiving training_config.json.
    """
    config_path = os.path.join(output_dir, "training_config.json")
    with open(config_path, "w", encoding = "utf-8") as file:
        json.dump(config, file, indent = 2, ensure_ascii = False, default = str)


def _is_non_empty_string(value: Any) -> bool:
    """Check whether a value is a non-empty string.

    Args:
        value: Value to validate.

    Returns:
        True when value contains non-whitespace text.
    """
    return isinstance(value, str) and bool(value.strip())


def _validate_messages(messages: Any, record_index: int, require_assistant: bool = True) -> bool:
    """Validate one SFT messages list.

    Args:
        messages: Raw messages value.
        record_index: Record index used in log messages.
        require_assistant: Whether at least one assistant message is required.

    Returns:
        Whether the messages are valid for SFT expansion.
    """
    if not isinstance(messages, list) or not messages:
        logger.error("Record %s: messages must be a non-empty list.", record_index)
        return False

    valid_roles = {"system", "user", "assistant"}
    assistant_count = 0
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            logger.error(
                "Record %s, message %s: expected an object.",
                record_index,
                message_index
            )
            return False
        if message.get("role") not in valid_roles:
            logger.error(
                "Record %s, message %s: invalid role.",
                record_index,
                message_index
            )
            return False
        if not _is_non_empty_string(message.get("content")):
            logger.error(
                "Record %s, message %s: content must be non-empty.",
                record_index,
                message_index
            )
            return False
        if message["role"] == "assistant":
            if message_index == 0:
                logger.error("Record %s: assistant message has no context.", record_index)
                return False
            assistant_count += 1

    if require_assistant and assistant_count == 0:
        logger.error("Record %s: no assistant message found.", record_index)
        return False

    return True


def _validate_history(history: Any, record_index: int) -> bool:
    """Validate optional DPO history.

    Args:
        history: Raw history value.
        record_index: Record index used in log messages.

    Returns:
        Whether history uses a supported representation.
    """
    if history in (None, []):
        return True
    if not isinstance(history, list):
        logger.error("Record %s: history must be a list.", record_index)
        return False
    if all(isinstance(item, dict) for item in history):
        return _validate_messages(
            history,
            record_index,
            require_assistant = False
        )

    for history_index, pair in enumerate(history):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            logger.error(
                "Record %s, history %s: expected a user-assistant pair.",
                record_index,
                history_index
            )
            return False
        if not all(_is_non_empty_string(content) for content in pair):
            logger.error(
                "Record %s, history %s: both messages must be non-empty.",
                record_index,
                history_index
            )
            return False

    return True


def validate_data_format(data_path: str, mode: str = "dpo") -> bool:
    """Validate every record in an SFT or DPO JSON dataset.

    Args:
        data_path: Path to the JSON dataset.
        mode: Dataset mode, either dpo or sft.

    Returns:
        Whether the complete dataset is valid.
    """
    if mode not in {"dpo", "sft"}:
        logger.error("Unsupported data validation mode: %s", mode)
        return False

    try:
        with open(data_path, "r", encoding = "utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        logger.error("Failed to read dataset %s: %s", data_path, error)
        return False

    if not isinstance(data, list) or not data:
        logger.error("Dataset must be a non-empty JSON array: %s", data_path)
        return False

    for record_index, record in enumerate(data):
        if not isinstance(record, dict):
            logger.error("Record %s must be an object.", record_index)
            return False

        if mode == "sft":
            if not _validate_messages(record.get("messages"), record_index):
                return False
            continue

        for field_name in ["instruction", "chosen", "rejected"]:
            if not _is_non_empty_string(record.get(field_name)):
                logger.error(
                    "Record %s: %s must be non-empty.",
                    record_index,
                    field_name
                )
                return False
        if "input" in record and not isinstance(record["input"], str):
            logger.error("Record %s: input must be a string.", record_index)
            return False
        if not _validate_history(record.get("history", []), record_index):
            return False

    logger.info("Validated %s dataset: %s", mode.upper(), data_path)
    return True


def _training_summary_lines(model_args, data_args, training_args) -> List[str]:
    """Build human-readable training summary lines.

    Args:
        model_args: ModelArguments instance.
        data_args: DataArguments instance.
        training_args: TrainingArguments instance.

    Returns:
        Ordered training summary lines.
    """
    lines = [
        "=" * 60,
        "Training Configuration",
        "=" * 60,
        f"model_path: {model_args.model_name_or_path}",
        f"train_data_path: {data_args.train_data_path}",
        f"eval_data_path: {data_args.eval_data_path or 'disabled'}",
        f"output_dir: {training_args.output_dir}",
        f"training_mode: {'DPO' if training_args.dpo else 'SFT'}",
        f"LoRA: {training_args.lora}",
        f"QLoRA: {training_args.qlora}",
        f"model_max_length: {training_args.model_max_length}",
        f"num_train_epochs: {training_args.num_train_epochs}",
        f"per_device_train_batch_size: {training_args.per_device_train_batch_size}",
        f"learning_rate: {training_args.learning_rate}"
    ]
    if training_args.dpo:
        lines.extend(
            [
                f"dpo_beta: {training_args.dpo_beta}",
                f"sft_loss_weight: {training_args.sft_loss_weight}",
                f"reference_model: {training_args.reference_model_path or 'main model'}"
            ]
        )
    lines.append("=" * 60)

    return lines


def print_training_summary(model_args, data_args, training_args) -> None:
    """Print the resolved training configuration.

    Args:
        model_args: ModelArguments instance.
        data_args: DataArguments instance.
        training_args: TrainingArguments instance.
    """
    for line in _training_summary_lines(model_args, data_args, training_args):
        print(line)


def logger_training_summary(model_args, data_args, training_args) -> None:
    """Log the resolved training configuration.

    Args:
        model_args: ModelArguments instance.
        data_args: DataArguments instance.
        training_args: TrainingArguments instance.
    """
    for line in _training_summary_lines(model_args, data_args, training_args):
        logger.info(line)
