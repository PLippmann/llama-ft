import os
import sys
from pathlib import Path

# Add src to Python path
src_path = str(Path(__file__).parent.parent / "src")
sys.path.append(src_path)

from utils.config_utils import load_config
from data.data_processor import DataProcessor
from training.trainer import LlamaTrainer

def main():
    # Load configuration
    config = load_config("config/default_config.yaml")
    
    # Process dataset
    data_processor = DataProcessor(config)
    dataset = data_processor.load_and_transform_dataset()
    
    # Train model
    trainer = LlamaTrainer(config)
    trainer.train(dataset)

if __name__ == "__main__":
    main() 