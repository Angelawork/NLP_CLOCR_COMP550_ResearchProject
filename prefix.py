import torch
from transformers import MistralConfig, MistralForCausalLM, AutoTokenizer
from peft import PrefixTuningConfig, get_peft_model, PeftModel
import pandas as pd
from datasets import Dataset

def get_model():
    # Configure a smaller Mistral model for testing
    model_config = MistralConfig(
        vocab_size=32000,
        hidden_size=512,
        max_position_embeddings=32768,
        num_attention_heads=16,
        num_hidden_layers=8,
        num_key_value_heads=4,
    )
    return MistralForCausalLM(model_config)

def prepare_dataset(csv_path, tokenizer, max_length=128):
    # Load dataset
    df = pd.read_csv(csv_path)
    
    def tokenize_function(examples):
        prompts = [f"Correct OCR: {text} -> {gt}" 
                  for text, gt in zip(examples["OCR Text"], examples["Ground Truth"])]
        
        return tokenizer(
            prompts,
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )
    
    # Create dataset
    dataset = Dataset.from_pandas(df)
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset

def train_model(model, dataset, num_epochs=3):
    from transformers import Trainer, TrainingArguments
    
    training_args = TrainingArguments(
        output_dir="results",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=4,
        save_steps=1000,
        save_total_limit=2,
        logging_steps=100,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    
    trainer.train()
    return model

def main():
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Get base model
    print("Initializing model...")
    model = get_model()
    
    # Configure prefix tuning
    peft_config = PrefixTuningConfig(
        task_type="CAUSAL_LM",
        num_virtual_tokens=30,
        prefix_projection=True
    )
    
    # Get PEFT model
    model = get_peft_model(model, peft_config)
    model.config.use_cache = False
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    # Prepare dataset
    print("Preparing dataset...")
    dataset = prepare_dataset("dataset/train.csv", tokenizer)
    
    # Train model
    print("Training model...")
    model = train_model(model, dataset)
    
    # Save model
    print("Saving model...")
    model.save_pretrained("ocr_prefix_tuned_mistral")
    
    # Test the model
    def correct_ocr(text):
        inputs = tokenizer(f"Correct OCR: {text} -> ", return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                num_beams=1
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True).split("->")[-1].strip()
    
    # Test examples
    test_texts = [
        "Ths is an exmple of OCR txt wth errors.",
        "The queck brown tox jumps over tho lazy dag."
    ]
    
    print("\nTesting model:")
    for text in test_texts:
        print(f"\nOriginal: {text}")
        print(f"Corrected: {correct_ocr(text)}")

if __name__ == "__main__":
    main()