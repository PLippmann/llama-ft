"""Data preprocessing script for LLaMA fine-tuning."""

from datasets import load_dataset
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def transform_conversation(example: dict, system_prompt: str = "") -> dict:
    """Transform Dolly dataset format to LLAMA 3 format.
    
    Args:
        example: Dictionary containing the example data
        system_prompt: Optional system prompt to prepend
        
    Returns:
        Dictionary with transformed text
    """
    reformatted_segments = []
    
    # Include system prompt if provided
    if system_prompt:
        reformatted_segments.append(f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n")
    
    # Format the instruction (context + instruction if context exists)
    instruction = example['instruction']
    if example['context']:
        instruction = f"Context: {example['context']}\n\nInstruction: {instruction}"
    
    # Format the response
    response = example['response']
    
    # Combine into LLAMA format
    conversation = f"<s>[INST] {instruction.strip()} [/INST] {response.strip()} </s>"
    reformatted_segments.append(conversation)

    return {'text': ''.join(reformatted_segments)}

def main():
    """Main function to process and upload the dataset."""
    logger.info("Loading Dolly dataset...")
    dataset = load_dataset('databricks/databricks-dolly-15k')

    # Shuffle the dataset and slice it
    dataset = dataset['train'].shuffle(seed=42)
    
    logger.info("Transforming dataset to LLaMA format...")
    transformed_dataset = dataset.map(transform_conversation)
    
    logger.info("Pushing transformed dataset to HuggingFace Hub...")
    transformed_dataset.push_to_hub("dolly-15k-llama3")
    logger.info("Dataset processing complete!")

if __name__ == "__main__":
    main() 