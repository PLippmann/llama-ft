from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    pipeline
)
from datasets import load_dataset
import torch
import os
import logging

# Set memory allocation strategy
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Configure logger
logger = logging.getLogger(__name__)

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
    training_args = TrainingArguments(
        output_dir="./results",
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
        hub_token=hf_token,  # Add the token here
        hub_model_id="curtsmith/llama3.2-3b-dolly-15k",  # Replace with your username
        hub_strategy="every_save",
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        max_grad_norm=0.3,
        max_steps=1,
    )
    
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

def load_model_and_tokenizer(model_id, hf_token):
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_auth_token=hf_token  # Changed from token=True
    )
    
    # Set up padding token
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_auth_token=hf_token,  # Changed from token=True
        low_cpu_mem_usage=True
    )
    
    # Enable gradient checkpointing after model initialization
    model.gradient_checkpointing_enable()
    
    # Make sure the model knows about the padding token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False  # Disable KV-cache for training

    return tokenizer, model

def prepare_dataset(tokenizer):
    # Load the processed dataset
    dataset = load_dataset("curtsmith/dolly-15k-llama3")
    
    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=4096,
            padding=False,
            return_tensors=None,
        )

    # Tokenize the dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        remove_columns=dataset["train"].column_names,
        num_proc=4,
    )

    return tokenized_dataset

def test_generation(model, tokenizer):
    # Run text generation pipeline with our next model
    prompt = "What is a large language model?"
    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,  # Add repetition penalty
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    result = pipe(f"{prompt}")
    print(result[0]['generated_text'])

if __name__ == "__main__":
    main() 