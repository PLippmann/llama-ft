import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Dict, Any, Tuple

def setup_model_and_tokenizer(config: Dict[str, Any]) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Setup the model and tokenizer with the specified configuration."""
    # Setup quantization config
    compute_dtype = getattr(torch, config['quantization']['compute_dtype'])
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config['quantization']['use_4bit'],
        bnb_4bit_quant_type=config['quantization']['quant_type'],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=config['quantization']['use_nested_quant'],
    )

    # Check GPU compatibility
    if compute_dtype == torch.float16 and config['quantization']['use_4bit']:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)

    # Load model with RoPE scaling configuration
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['name'],
        quantization_config=bnb_config,
        device_map={"": 0},
        rope_scaling=config['model']['rope_scaling']
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['name'],
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer 