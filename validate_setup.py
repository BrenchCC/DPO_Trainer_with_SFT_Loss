"""Validate the local environment and project structure without loading models."""

import os
import sys
import logging
import importlib
from typing import List, Tuple

sys.path.append(os.getcwd())

from utils.tools import validate_data_format

logger = logging.getLogger(__name__)


def check_python_version() -> bool:
    """Check the minimum supported Python version.

    Returns:
        Whether the current interpreter is supported.
    """
    supported = sys.version_info >= (3, 10)
    if supported:
        logger.info("Python version: %s", sys.version.split()[0])
    else:
        logger.error("Python 3.10 or newer is required.")

    return supported


def check_dependencies() -> bool:
    """Check required Python packages without checking optional QLoRA support.

    Returns:
        Whether every required package can be imported.
    """
    required_packages = [
        "torch",
        "transformers",
        "peft",
        "accelerate"
    ]
    missing_packages = []
    for package_name in required_packages:
        try:
            module = importlib.import_module(package_name)
            logger.info(
                "Dependency: %s %s",
                package_name,
                getattr(module, "__version__", "unknown")
            )
        except ImportError:
            logger.error("Missing dependency: %s", package_name)
            missing_packages.append(package_name)

    if importlib.util.find_spec("bitsandbytes") is None:
        logger.info("Optional QLoRA dependency bitsandbytes is not installed.")

    return not missing_packages


def check_project_structure() -> bool:
    """Check files required by the complete training workflow.

    Returns:
        Whether every required project file exists.
    """
    required_files = [
        "config/arguments.py",
        "data/datasets.py",
        "models/model_loader.py",
        "trainers/dpo_trainer.py",
        "utils/tools.py",
        "scripts/train_dpo.sh",
        "scripts/train_sft.sh",
        "data_examples/dpo_data_example.json",
        "data_examples/sft_data_example.json",
        "train.py"
    ]
    missing_files = []
    for file_path in required_files:
        if os.path.isfile(file_path):
            logger.info("Project file: %s", file_path)
        else:
            logger.error("Missing project file: %s", file_path)
            missing_files.append(file_path)

    return not missing_files


def check_example_data() -> bool:
    """Validate bundled SFT and DPO example data.

    Returns:
        Whether both example files pass complete validation.
    """
    dpo_valid = validate_data_format(
        "data_examples/dpo_data_example.json",
        "dpo"
    )
    sft_valid = validate_data_format(
        "data_examples/sft_data_example.json",
        "sft"
    )

    return dpo_valid and sft_valid


def check_module_interfaces() -> bool:
    """Import required modules and verify their public symbols.

    Returns:
        Whether every required interface can be imported.
    """
    module_interfaces: List[Tuple[str, List[str]]] = [
        (
            "config.arguments",
            ["ModelArguments", "DataArguments", "TrainingArguments"]
        ),
        (
            "data.datasets",
            ["SupervisedDataset", "DPODataset", "DPODataCollator"]
        ),
        (
            "models.model_loader",
            ["load_model_and_tokenizer", "load_reference_model"]
        ),
        ("trainers.dpo_trainer", ["DPOTrainer"]),
        ("utils.tools", ["validate_data_format", "logger_training_summary"])
    ]

    for module_name, symbol_names in module_interfaces:
        try:
            module = importlib.import_module(module_name)
        except Exception as error:
            logger.error("Failed to import %s: %s", module_name, error)
            return False
        for symbol_name in symbol_names:
            if not hasattr(module, symbol_name):
                logger.error("Missing interface: %s.%s", module_name, symbol_name)
                return False
        logger.info("Module interface: %s", module_name)

    return True


def main() -> int:
    """Run every setup check and return a process exit code.

    Returns:
        Zero when every check succeeds, otherwise one.
    """
    checks = [
        check_python_version(),
        check_dependencies(),
        check_project_structure(),
        check_example_data(),
        check_module_interfaces()
    ]
    if all(checks):
        logger.info("All setup checks passed.")
        return 0

    logger.error("One or more setup checks failed.")
    return 1


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()]
    )
    raise SystemExit(main())
