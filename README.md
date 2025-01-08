# LLaMA-FT: Fine-tuning Llama 3 Models

A streamlined repository for fine-tuning Llama 3 models for my own use. This project provides tools for data preprocessing and an efficient training script using HuggingFace.

## Features

- Data preprocessing pipeline for dataset with Llama 3 chat format
- Fine-tuning script for LLaMA 3 models
- Memory-efficient training with gradient checkpointing
- Automatic model pushing to Hugging Face Hub

## Installation

```bash
git clone https://github.com/yourusername/llama-ft.git && cd llama-ft
pip install -r requirements.txt
```
Make sure to set the HF_TOKEN environment variable.

## Usage

### 1. Data Preprocessing

To preprocess the Dolly dataset:

```bash
python llama_ft/data/process.py
```

This will:
- Load the specified dataset from HuggingFace
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

### 3. Inference

To generate text with the fine-tuned model:

```bash
python llama_ft/inference/generate.py
```

## Configuration

I run this on a single A100 80GB.

## Requirements

See `requirements.txt` for full dependencies.

## Project Structure

```
llama-ft/
├── llama_ft/
│   ├── data/
│   │   └── process.py
│   ├── train/
│   │   └── finetune.py
│   └── inference/
│       └── generate.py
├── requirements.txt
└── README.md
```
