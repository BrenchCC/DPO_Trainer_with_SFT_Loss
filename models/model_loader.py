"""Model and tokenizer loading helpers."""

import warnings
from typing import Optional, Tuple
from importlib.util import find_spec

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def _resolve_dtype(bf16: bool, fp16: bool) -> torch.dtype:
    """Resolve the requested model dtype.

    Args:
        bf16: Whether to load with bfloat16.
        fp16: Whether to load with float16.

    Returns:
        Resolved torch dtype.
    """
    if bf16 and fp16:
        raise ValueError("bf16 and fp16 cannot both be enabled.")
    if bf16:
        return torch.bfloat16
    if fp16:
        return torch.float16

    return torch.float32


def load_model_and_tokenizer(
    model_name_or_path: str,
    max_length: int = 512,
    lora: bool = False,
    qlora: bool = False,
    bf16: bool = False,
    fp16: bool = False,
    cache_dir: Optional[str] = None
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a causal language model and its tokenizer.

    Args:
        model_name_or_path: Local path or Hugging Face model ID.
        max_length: Maximum tokenizer sequence length.
        lora: Whether to attach LoRA adapters.
        qlora: Whether to load the base model with 4-bit quantization.
        bf16: Whether to use bfloat16 model weights.
        fp16: Whether to use float16 model weights.
        cache_dir: Optional Hugging Face cache directory.

    Returns:
        Loaded model and tokenizer.
    """
    if max_length < 2:
        raise ValueError("max_length must be at least 2.")
    if qlora and not lora:
        raise ValueError("lora must be enabled when qlora is enabled.")
    if qlora and not torch.cuda.is_available():
        raise RuntimeError("QLoRA requires a CUDA device.")
    if qlora and find_spec("bitsandbytes") is None:
        raise ImportError("QLoRA requires the optional bitsandbytes package.")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir = cache_dir,
        trust_remote_code = True
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("The tokenizer must define an EOS token.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = max_length

    dtype = _resolve_dtype(bf16, fp16)
    model_kwargs = {
        "cache_dir": cache_dir,
        "dtype": dtype,
        "trust_remote_code": True
    }

    if qlora:
        compute_dtype = dtype if dtype != torch.float32 else torch.float16
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit = True,
            bnb_4bit_compute_dtype = compute_dtype,
            bnb_4bit_quant_type = "nf4",
            bnb_4bit_use_double_quant = True
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **model_kwargs
    )

    if lora:
        from peft import LoraConfig, TaskType, get_peft_model
        from peft import prepare_model_for_kbit_training

        if qlora:
            model = prepare_model_for_kbit_training(model)

        target_modules = ["q_proj", "v_proj"]
        if getattr(model.config, "architectures", None) == ["MiniCPM3ForCausalLM"]:
            target_modules = [
                "q_a_proj",
                "kv_a_proj_with_mqa",
                "q_b_proj",
                "kv_b_proj"
            ]

        lora_config = LoraConfig(
            init_lora_weights = "gaussian",
            task_type = TaskType.CAUSAL_LM,
            target_modules = target_modules,
            r = 64,
            lora_alpha = 32,
            lora_dropout = 0.1,
            inference_mode = False
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        model.enable_input_require_grads()

    return model, tokenizer


def load_model_and_toknizer(
    model_name_or_path: str,
    max_length: int = 512,
    lora: bool = False,
    qlora: bool = False,
    bf16: bool = False,
    fp16: bool = False,
    cache_dir: Optional[str] = None
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Call load_model_and_tokenizer through the deprecated misspelling.

    Args:
        model_name_or_path: Local path or Hugging Face model ID.
        max_length: Maximum tokenizer sequence length.
        lora: Whether to attach LoRA adapters.
        qlora: Whether to load the base model with 4-bit quantization.
        bf16: Whether to use bfloat16 model weights.
        fp16: Whether to use float16 model weights.
        cache_dir: Optional Hugging Face cache directory.

    Returns:
        Loaded model and tokenizer.
    """
    warnings.warn(
        "load_model_and_toknizer is deprecated; use load_model_and_tokenizer.",
        DeprecationWarning,
        stacklevel = 2
    )

    return load_model_and_tokenizer(
        model_name_or_path,
        max_length,
        lora,
        qlora,
        bf16,
        fp16,
        cache_dir
    )


def load_reference_model(
    model_path: str,
    bf16: bool = False,
    fp16: bool = False,
    cache_dir: Optional[str] = None
) -> AutoModelForCausalLM:
    """Load and freeze the DPO reference model.

    Args:
        model_path: Local path or Hugging Face model ID.
        bf16: Whether to use bfloat16 model weights.
        fp16: Whether to use float16 model weights.
        cache_dir: Optional Hugging Face cache directory.

    Returns:
        Frozen reference model in evaluation mode.
    """
    reference_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        cache_dir = cache_dir,
        dtype = _resolve_dtype(bf16, fp16),
        trust_remote_code = True
    )
    reference_model.eval()
    for parameter in reference_model.parameters():
        parameter.requires_grad = False

    return reference_model
