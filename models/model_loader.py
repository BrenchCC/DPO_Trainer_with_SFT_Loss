import torch
from typing import Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def load_model_and_toknizer(
        model_name_or_path: str,
        max_length: int = 8192,
        lora: bool = True,
        qlora: bool = False,
        bf16: bool = False,
        fp16: bool = False,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load model and tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_mode = True)
    tokenizer.pad_token = tokenizer.eos_token

    # Ensure don't use bf16 or fp16 at the same time
    assert not (bf16 and fp16), "bf16 or fp16, can't not use both"
    if bf16:
        dtype = torch.bfloat16
    elif fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32

    if qlora:
        assert lora, "lora option must be True when use qlora is True"
        quantizaiton_config = BitsAndBytesConfig(
            lora_in_4bit = True,
            load_in_8bit = False,
            bnb_4bit_compute_dtype = torch.float16,
            bnb_4bit_quant_storage = torch.uint8,
            bnb_4bit_quant_type = "nf4",
            bnb_4bit_use_double_quant = True,
            llm_int8_enable_fp32_cpu_offload = False,
            llm_int8_has_fp16_weight = False,
            llm_int8_threshold = 6.0
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            trust_remote_mode = True,
            quantization_config = quantizaiton_config
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype = dtype,
            trust_remote_mode = True,
        )

    if lora:
        from peft import LoraConfig, TaskType, get_peft_model
        lora_config = LoraConfig(
            init_lora_weights = "gaussian",
            task_type = TaskType.CAUSAL_LM,
            target_modules = (
                [
                    "q_a_proj",
                    "kv_a_proj_with_mqa",
                    "q_b_proj",
                    "kv_b_proj"
                ]
                # for MiniCPM3
                if hasattr(model.config, "architectures") and model.config.architectures == ["MiniCPM3ForCausalLM"]
                # for other models
                else ["q_proj", "v_proj"]
            ),
            r = 64,
            lora_alpha = 32,
            lora_dropout = 0.1,
            inference_mode = False,
        )
        model = get_peft_model(model, lora_config)
        model.print_tainable_parameters()
        model.enable_input_require_grads()

    return model, tokenizer

def load_reference_model(model_path: str, bf16: bool = False, fp16: bool = False) -> AutoModelForCausalLM:
    """
    Load reference model for dpo training
    """
    assert not (bf16 and fp16), "bf16 or fp16, not both"
    if bf16:
        dtype = torch.bfloat16
    elif fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32

    reference_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    # reference model don't need to calculate gradients
    for param in reference_model.parameters():
        param.requires_grad = False

    return reference_model