from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_id: str = "curtsmith/llama3.2-3b-dolly-15k"):
    """Load the fine-tuned model and tokenizer."""
    logger.info(f"Loading model and tokenizer from {model_id}...")
    
    try:
        # First try loading the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        logger.info("Successfully loaded tokenizer")
        
        # Then load the model
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        logger.info("Successfully loaded model")
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Error loading model or tokenizer: {str(e)}")
        logger.error("Please ensure both model and tokenizer were properly saved to the Hub")
        raise

def setup_pipeline(model, tokenizer):
    """Create a text generation pipeline with the model."""
    return pipeline(
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

def generate_responses(pipe, prompts):
    """Generate responses for a list of prompts."""
    for i, prompt in enumerate(prompts, 1):
        logger.info(f"\nPrompt {i}: {prompt}")
        
        # Format prompt with Llama 2 instruction format
        formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        
        # Generate response
        result = pipe(formatted_prompt)
        response = result[0]['generated_text']
        
        # Clean up response by removing the prompt
        response = response.split('[/INST]')[-1].strip()
        
        logger.info(f"Response {i}: {response}\n")
        logger.info("-" * 80)

def main():
    # List of test prompts
    test_prompts = [
        "What is a large language model?",
        "Explain the concept of fine-tuning in machine learning.",
        "Write a short poem about artificial intelligence.",
        "What are the ethical considerations when developing AI?",
        "How do neural networks learn from data?"
    ]
    
    # Load model and create pipeline
    model, tokenizer = load_model()
    pipe = setup_pipeline(model, tokenizer)
    
    # Generate responses
    logger.info("Generating responses...")
    generate_responses(pipe, test_prompts)

if __name__ == "__main__":
    main() 