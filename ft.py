from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import torch

def main():
    # Load model and tokenizer
    model_id = "meta-llama/Llama-3.2-3b-instruct"  # The official Llama 3.2 instruct model
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=True  # You'll need your HF token with Meta license
    )
    
    # Set up padding token
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        token=True  # You'll need your HF token with Meta license
    )
    # Make sure the model knows about the padding token
    model.config.pad_token_id = tokenizer.pad_token_id

    # Load the processed dataset
    dataset = load_dataset("your-username/dolly-15k-llama3")  # Replace with your actual HF username
    
    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=2048,
            padding=False,
            return_tensors=None,
        )

    # Tokenize the dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        remove_columns=dataset["train"].column_names,
        num_proc=4,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=500,
        logging_steps=100,
        save_steps=1000,
        save_total_limit=3,
        fp16=True,
        push_to_hub=True,
        hub_model_id="dolly-15k-llama3-ft"
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    # Start training
    trainer.train()
    
    # Save the model
    trainer.save_model()
    
    # Push to Hub if specified in training arguments
    if training_args.push_to_hub:
        trainer.push_to_hub()

if __name__ == "__main__":
    main() 