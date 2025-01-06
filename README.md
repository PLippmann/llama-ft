# LLaMA-FT: Fine-tuning LLaMA Models

A streamlined repository for fine-tuning LLaMA models using the Dolly dataset. This project provides tools for data preprocessing and model fine-tuning with efficient training configurations.

## Features

- Data preprocessing pipeline for Dolly dataset
- Fine-tuning script for LLaMA models
- Memory-efficient training with gradient checkpointing
- Automatic model pushing to Hugging Face Hub

## Installation

```bash
git clone https://github.com/yourusername/llama-ft.git
cd llama-ft
pip install -r requirements.txt
```

## Usage

### 1. Data Preprocessing

To preprocess the Dolly dataset:

```bash
python llama_ft/data/process.py
```

This will:
- Load the Databricks Dolly dataset
- Transform it into LLaMA chat format
- Push the processed dataset to HuggingFace Hub

### 2. Fine-tuning

To start the fine-tuning process:

```bash
python llama_ft/train/finetune.py
```

The script includes:
- Automatic model loading with correct configurations
- Memory-efficient training setup
- Gradient checkpointing and mixed precision training
- Regular model checkpointing
- Automatic upload to HuggingFace Hub

## Configuration

I ran this on a single A100 80GB.

The training configuration includes:
- Learning rate: 2e-5
- Training epochs: 3
- Batch size: 2 with gradient accumulation of 16
- BF16 mixed precision
- Gradient checkpointing enabled
- Cosine learning rate scheduler

## Requirements

See `requirements.txt` for full dependencies. Key requirements:
- transformers >= 4.38.0
- torch >= 2.2.0
- datasets >= 2.16.0
- accelerate >= 0.27.0

## Project Structure

```
llama-ft/
├── llama_ft/
│   ├── data/
│   │   └── process.py
│   └── train/
│       └── finetune.py
├── requirements.txt
└── README.md
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.