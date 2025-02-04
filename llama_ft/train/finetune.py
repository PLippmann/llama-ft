"""Fine-tuning script for LLaMA models."""

import os
import logging
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    pipeline
)
from datasets import load_dataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set memory allocation strategy
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def load_model_and_tokenizer(model_id: str, hf_token: str):
    """Load the model and tokenizer.
    
    Args:
        model_id: HuggingFace model identifier
        hf_token: HuggingFace API token
        
    Returns:
        tuple: (tokenizer, model)
    """
    logger.info(f"Loading tokenizer from {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=hf_token
    )
    
    # Set up padding token
    tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Loading model from {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
        low_cpu_mem_usage=True
    )
    
    # Enable gradient checkpointing and configure model
    model.gradient_checkpointing_enable()
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False  # Disable KV-cache for training
    
    # Ensure parameters require gradients
    for param in model.parameters():
        param.requires_grad = True
    
    return tokenizer, model

def prepare_dataset(tokenizer, dataset_name: str = "curtsmith/dolly-15k-llama3"):
    """Prepare and tokenize the dataset.
    
    Args:
        tokenizer: The tokenizer to use
        dataset_name: Name of the dataset on HuggingFace Hub
        
    Returns:
        tokenized dataset
    """
    logger.info(f"Loading dataset {dataset_name}...")
    dataset = load_dataset(dataset_name)
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=4096,
            padding=False,
            return_tensors=None,
        )

    logger.info("Tokenizing dataset...")
    return dataset.map(
        tokenize_function,
        remove_columns=dataset["train"].column_names,
        num_proc=4,
    )

def get_training_args(output_dir: str = "./results", hub_model_id: str = "llama3.2-3b-dolly-15k", hf_token: str = None):
    """Get training arguments.
    
    Args:
        output_dir: Directory to save the model
        hub_model_id: Model ID for HuggingFace Hub
        hf_token: HuggingFace API token
        
    Returns:
        TrainingArguments
    """
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=100,
        save_steps=1000,
        save_total_limit=3,
        bf16=True,
        push_to_hub=True,
        hub_token=hf_token,
        hub_model_id=hub_model_id,
        hub_strategy="every_save",
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        max_grad_norm=0.3,
    )

def test_generation(model, tokenizer, prompt: str = "What is a large language model?"):
    """Test the model with a sample prompt.
    
    Args:
        model: The trained model
        tokenizer: The tokenizer
        prompt: Test prompt
    """
    logger.info("Testing model generation...")
    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    result = pipe(f"<s>[INST] {prompt} [/INST]")
    logger.info(f"Generated text: {result[0]['generated_text']}")

def main():
    """Main training function."""
    # Get HuggingFace token from environment variable
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("Please set the HF_TOKEN environment variable")

    # Initialize model and tokenizer
    model_id = "meta-llama/Llama-3.2-3b-instruct"
    tokenizer, model = load_model_and_tokenizer(model_id, hf_token)
    
    # Prepare dataset
    tokenized_dataset = prepare_dataset(tokenizer)
    
    # Get training arguments with token
    training_args = get_training_args(hub_model_id="your-username/llama3.2-3b-dolly-15k", hf_token=hf_token)
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    # Save both model and tokenizer
    logger.info("Saving model and tokenizer...")
    output_dir = training_args.output_dir
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Push to Hub if specified
    if training_args.push_to_hub:
        logger.info("Pushing model and tokenizer to HuggingFace Hub...")
        model.push_to_hub(training_args.hub_model_id, token=hf_token)
        tokenizer.push_to_hub(training_args.hub_model_id, token=hf_token)

    # Test generation
    test_generation(model, tokenizer)

if __name__ == "__main__":
    main() 