from transformers import TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer
from typing import Dict, Any
import torch

from .model_utils import setup_model_and_tokenizer

class LlamaTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model, self.tokenizer = setup_model_and_tokenizer(config)
        
    def train(self, dataset):
        """Train the model on the provided dataset."""
        # Setup LoRA configuration
        peft_config = LoraConfig(
            lora_alpha=self.config['lora']['alpha'],
            lora_dropout=self.config['lora']['dropout'],
            r=self.config['lora']['r'],
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Setup training arguments
        training_args = TrainingArguments(
            **{k: v for k, v in self.config['training'].items() 
               if k not in ['max_seq_length', 'packing']}
        )

        # Initialize trainer
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=dataset,
            peft_config=peft_config,
            dataset_text_field=self.config['dataset']['text_field'],
            max_seq_length=self.config['training']['max_seq_length'],
            tokenizer=self.tokenizer,
            args=training_args,
            packing=self.config['training']['packing'],
        )

        # Train model
        trainer.train()
        
        # Save trained model
        trainer.model.save_pretrained(self.config['model']['new_model_name'])
        
        return trainer 