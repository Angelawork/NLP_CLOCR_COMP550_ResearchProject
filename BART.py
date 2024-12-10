from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    BartForConditionalGeneration,
    BartTokenizer
)
from peft import (
    get_peft_model,
    PrefixTuningConfig,
    PromptTuningConfig,
    TaskType,
    PromptTuningInit
)
import numpy as np
import argparse
import os
import pandas as pd
import torch
import yaml
from Levenshtein import distance as levenshtein_distance
from huggingface_hub import login
from main import calculate_cer, calculate_wer
# Environment setup
hf_api_key = os.getenv("HF_API_KEY")
login(token=hf_api_key)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

def prepare_input_text(text):
    """Clean and standardize input text."""
    return text.strip()


def load_config(file):
    with open(file, "r") as f:
        config = yaml.safe_load(f)
    
    bart_config = config.get("bart", {})
    
    numeric_fields = {
        "learning_rate": float,
        "num_train_epochs": int,
        "per_device_train_batch_size": int,
        "per_device_eval_batch_size": int,
        "gradient_accumulation_steps": int,
        "warmup_steps": int,
        "weight_decay": float,
        "logging_steps": int,
        "eval_steps": int,
        "save_steps": int,
        "save_total_limit": int,
        "num_virtual_tokens": int,
        "encoder_hidden_size": int
    }
    
    for field, type_func in numeric_fields.items():
        if field in bart_config:
            bart_config[field] = type_func(bart_config[field])
    
    return bart_config

def prepare_dataset(data, tokenizer, max_length=512):
    def preprocess_function(examples):
        cleaned_inputs = [f"OCR: {text} </s>" for text in examples["OCR Text"]]
        
        model_inputs = tokenizer(
            cleaned_inputs,
            max_length=max_length,
            padding=False,
            truncation=True
        )
        
        # Format targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["Ground Truth"],
                max_length=max_length,
                padding=False,
                truncation=True
            )
            
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    return data.map(preprocess_function, batched=True)

def main(args):
    config = load_config(args.config)
    model_name = args.model
    output_dir = os.path.join("model", f"bart-ocr-{args.tuning}")
    
    # Load and prepare data
    train_df = pd.read_csv(args.data)
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    split_idx = int(len(train_df) * 0.9)
    train_df_split = train_df[:split_idx]
    eval_df_split = train_df[split_idx:]
    
    train_dataset = Dataset.from_pandas(train_df_split)
    eval_dataset = Dataset.from_pandas(eval_df_split)
    
    default_args = {
        "evaluation_strategy": "steps",
        "eval_steps": 500,
        "learning_rate": 2e-5,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "weight_decay": 0.01,
        "save_total_limit": 3,
        "load_best_model_at_end": True,
        "num_train_epochs": 3,
        "save_strategy": "steps",
        "save_steps": 500
    }
    
    training_args_dict = {**default_args, **config}
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        **training_args_dict
    )
    
    # Initialize tokenizer and model
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    
    if args.tuning == "ptuning":
        peft_config = PromptTuningConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            prompt_tuning_init=PromptTuningInit.TEXT,
            prompt_tuning_init_text="Fix OCR errors",
            num_virtual_tokens=50, 
            tokenizer_name_or_path=model_name,
        )
        model = get_peft_model(model, peft_config)
    
    # Prepare datasets
    tokenized_train_dataset = prepare_dataset(train_dataset, tokenizer)
    tokenized_eval_dataset = prepare_dataset(eval_dataset, tokenizer)
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
    )
    
    # Train and save
    trainer.train()
    trainer.save_model(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P-tuning BART for OCR correction")
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/bart-base",
        help="BART model to use (e.g., facebook/bart-base, facebook/bart-large)"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file",
        default="./config.yaml"
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Path to training data",
        default="./dataset/train.csv"
    )
    parser.add_argument(
        "--tuning",
        type=str,
        choices=["ptuning"],
        default="ptuning",
        help="Specify tuning type"
    )
    
    args = parser.parse_args()
    main(args)