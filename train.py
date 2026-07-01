"""Command-line entry point for SFT, DPO, and DPO with SFT loss."""

import os
import sys
import logging
from dataclasses import asdict

import transformers
from transformers import DataCollatorForSeq2Seq, Trainer

sys.path.append(os.getcwd())

from trainers.dpo_trainer import DPOTrainer
from utils.tools import create_output_dir, logger_training_summary
from utils.tools import save_training_config, validate_data_format
from data.datasets import DPODataset, DPODataCollator, SupervisedDataset
from config.arguments import DataArguments, ModelArguments, TrainingArguments
from models.model_loader import load_model_and_tokenizer, load_reference_model

logger = logging.getLogger(__name__)


def parse_args():
    """Parse model, data, and training command-line arguments.

    Returns:
        Parsed ModelArguments, DataArguments, and TrainingArguments instances.
    """
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )

    return parser.parse_args_into_dataclasses()


def main() -> None:
    """Run SFT or DPO training from parsed command-line arguments."""
    model_args, data_args, training_args = parse_args()
    if not training_args.output_dir:
        raise ValueError("--output_dir is required.")
    if training_args.dpo:
        training_args.remove_unused_columns = False

    create_output_dir(training_args.output_dir)
    logger_training_summary(model_args, data_args, training_args)

    data_mode = "dpo" if training_args.dpo else "sft"
    if not validate_data_format(data_args.train_data_path, data_mode):
        raise ValueError("Training data validation failed.")
    if data_args.eval_data_path and not validate_data_format(
        data_args.eval_data_path,
        data_mode
    ):
        raise ValueError("Evaluation data validation failed.")

    logger.info("Loading model and tokenizer.")
    model, tokenizer = load_model_and_tokenizer(
        model_args.model_name_or_path,
        training_args.model_max_length,
        training_args.lora,
        training_args.qlora,
        training_args.bf16,
        training_args.fp16,
        training_args.cache_dir
    )

    save_training_config(
        {
            "model_args": asdict(model_args),
            "data_args": asdict(data_args),
            "training_args": training_args.to_dict()
        },
        training_args.output_dir
    )

    if training_args.dpo:
        reference_model_path = (
            training_args.reference_model_path or model_args.model_name_or_path
        )
        logger.info("Loading reference model: %s", reference_model_path)
        reference_model = load_reference_model(
            reference_model_path,
            training_args.bf16,
            training_args.fp16,
            training_args.cache_dir
        )

        train_dataset = DPODataset(
            data_args.train_data_path,
            tokenizer,
            training_args.model_max_length
        )
        eval_dataset = None
        if data_args.eval_data_path:
            eval_dataset = DPODataset(
                data_args.eval_data_path,
                tokenizer,
                training_args.model_max_length
            )

        trainer = DPOTrainer(
            model = model,
            args = training_args,
            train_dataset = train_dataset,
            eval_dataset = eval_dataset,
            processing_class = tokenizer,
            data_collator = DPODataCollator(tokenizer),
            reference_model = reference_model,
            dpo_beta = training_args.dpo_beta,
            sft_loss_weight = training_args.sft_loss_weight
        )
    else:
        train_dataset = SupervisedDataset(
            data_args.train_data_path,
            tokenizer,
            training_args.model_max_length
        )
        eval_dataset = None
        if data_args.eval_data_path:
            eval_dataset = SupervisedDataset(
                data_args.eval_data_path,
                tokenizer,
                training_args.model_max_length
            )

        data_collator = DataCollatorForSeq2Seq(
            tokenizer = tokenizer,
            model = model,
            padding = True,
            label_pad_token_id = -100,
            return_tensors = "pt"
        )
        trainer = Trainer(
            model = model,
            args = training_args,
            train_dataset = train_dataset,
            eval_dataset = eval_dataset,
            processing_class = tokenizer,
            data_collator = data_collator
        )

    logger.info("Starting training.")
    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    logger.info("Training completed: %s", training_args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()]
    )
    main()
