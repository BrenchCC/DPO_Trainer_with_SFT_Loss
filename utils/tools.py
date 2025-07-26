"""
Tools Function Module
"""
import os
import json
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_output_dir(output_dir: str) -> None:
    """
    Create a directory if it doesn't exist already.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)


def save_training_config(config: Dict[str, Any], output_dir: str) -> None:
    """
    Save the training configuration to a json file.
    """
    config_path = os.path.join(output_dir, "training_config.json")
    with open(config_path, "w", encoding = "utf-8") as f:
        json.dump(config, f, indent = 2, ensure_ascii = False)


def validate_data_format(data_path: str, mode: str = "dpo") -> bool:
    """
    Validate the data format.
    """
    try:
        with open(data_path, "r", encoding = "utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list) or len(data) == 0:
            logger.info(f"Error: {data_path} is not a valid data format or empty.")
            return False

        sample_data = data[0]
        if mode == "dpo":
            required_fields = ["instruction", "chosen", "rejected"]
            for field in required_fields:
                if field not in sample_data:
                    logger.info(f"Error: DPO data--{data_path} lacks required field {field}.")
        elif mode == "sft":
            if "messages" not in sample_data:
                logger.info(f"Error: SFT data--{data_path} lacks required field messages.")
                return False

        logger.info(f"Has Validated data format: {data_path}. Pass!")
    except Exception as e:
        logger.info(f"Failed to validate data format: {e}")

def print_training_summary(model_args, data_args, training_args) -> None:
    """
    print training configs summary
    """
    print("=" * 50)
    print("Training Configs Summary:")
    print("=" * 50)
    print(f"model_path: {model_args.model_name_or_path}")
    print(f"train_data_path: {data_args.train_data_path}")
    print(f"eval_data_path: {data_args.eval_data_path}")
    print(f"output_dir: {data_args.output_dir}")
    print(f"training_mode: {'DPO' if training_args.dpo else'SFT'}")
    print(f"LoRA: {training_args.lora}")
    print(f"QLoRA: {training_args.qlora}")
    print(f"max_seq_length: {training_args.model_max_length}")
    print(f"num_of_training_epochs: {training_args.num_train_epochs}")
    print(f"per_device_batch_size: {training_args.per_device_train_batch_size}")
    print(f"learning_rate: {training_args.learning_rate}")

    if training_args.dpo:
        print(f"DPO_Beta: {training_args.dpo_beta}")
        print(f"SFT_Loss_Weight: {training_args.sft_loss_weight}")
        print(f"Reference_Model: {training_args.reference_model_path or 'using main model'}")

    print("=" * 50)

def logger_training_summary(model_args, data_args, training_args) -> None:
    """
    logs of training configs summary
    """
    logger.info("*"*50)
    logger.info("Training Configs Summary:")
    logger.info("*" * 50)
    logger.info(f"model_path: {model_args.model_name_or_path}")
    logger.info(f"train_data_path: {data_args.train_data_path}")
    logger.info(f"eval_data_path: {data_args.eval_data_path}")
    logger.info(f"output_dir: {data_args.output_dir}")
    logger.info(f"training_mode: {'DPO' if training_args.dpo else'SFT'}")
    logger.info(f"LoRA: {training_args.lora}")
    logger.info(f"QLoRA: {training_args.qlora}")
    logger.info(f"max_seq_length: {training_args.model_max_length}")
    logger.info(f"num_of_training_epochs: {training_args.num_train_epochs}")
    logger.info(f"per_device_batch_size: {training_args.per_device_train_batch_size}")
    logger.info(f"learning_rate: {training_args.learning_rate}")

    if training_args.dpo:
        logger.info(f"DPO_Beta: {training_args.dpo_beta}")
        logger.info(f"SFT_Loss_Weight: {training_args.sft_loss_weight}")
        logger.info(f"Reference_Model: {training_args.reference_model_path or 'using main model'}")

    logger.info("*" * 50)