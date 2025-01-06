from datasets import load_dataset
from typing import Dict, Any

class DataProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def load_and_transform_dataset(self):
        """Load and transform the dataset according to the configuration."""
        dataset = load_dataset(self.config['dataset']['name'])
        
        # Shuffle and prepare the dataset
        dataset = dataset['train'].shuffle(seed=self.config['dataset']['seed'])
        
        # Transform the dataset
        transformed_dataset = dataset.map(self._transform_dolly_to_llama3)
        return transformed_dataset
    
    def _transform_dolly_to_llama3(self, example: Dict[str, str]) -> Dict[str, str]:
        """Transform Dolly format to Llama 3 instruction format.
        
        Dolly format:
        {
            'instruction': str,  # The instruction/question
            'context': str,     # Optional context/reference text
            'response': str,    # The response/answer
            'category': str     # Type of instruction
        }
        
        Llama 3 format:
        <|im_start|>system
        You are a helpful assistant.
        <|im_end|>
        <|im_start|>user
        {instruction}
        <|im_end|>
        <|im_start|>assistant
        {response}
        <|im_end|>
        """
        # Combine instruction with context if available
        instruction = example['instruction']
        if example.get('context') and example['context'].strip():
            instruction = f"{instruction}\n\nContext: {example['context']}"
            
        # Format in Llama 3 style
        formatted_text = (
            "<|im_start|>system\n"
            "You are a helpful assistant.\n"
            "<|im_end|>\n"
            f"<|im_start|>user\n{instruction.strip()}\n<|im_end|>\n"
            f"<|im_start|>assistant\n{example['response'].strip()}\n<|im_end|>"
        )
        
        return {'text': formatted_text} 