from datasets import load_dataset
import re

# Load the Dolly dataset
dataset = load_dataset('databricks/databricks-dolly-15k')

# Shuffle the dataset and slice it
dataset = dataset['train'].shuffle(seed=42)

def transform_conversation(example, system_prompt=""):
    """Transform Dolly dataset format to LLAMA 3 format."""
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

# Apply the transformation
transformed_dataset = dataset.map(transform_conversation)

# Push to hub with a appropriate name
transformed_dataset.push_to_hub("dolly-15k-llama3")
