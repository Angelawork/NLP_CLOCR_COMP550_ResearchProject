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
import numpy as np
import argparse
import os
import pandas as pd
import torch
import yaml
from Levenshtein import distance as levenshtein_distance
from huggingface_hub import login


# Environment setup
hf_api_key = os.getenv("HF_API_KEY")
login(token=hf_api_key)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

def test_prompt_template(example):
    return f"""Fix the OCR errors in the following text: {example}"""

def calculate_cer(prediction, ground_truth):
    # Ensure both strings are lowercase and strip whitespace
    prediction = prediction.lower().strip()
    ground_truth = ground_truth.lower().strip()
    
    # Calculate Levenshtein distance
    distance = levenshtein_distance(prediction, ground_truth)
    
    # Calculate CER
    cer = distance / len(ground_truth) if len(ground_truth) > 0 else 0
    return cer

def evaluate(tokenizer, model, data, output_file, batch_size=4):
    model.eval()
    preds, cer_values = [], []
    
    for i in range(0, len(data), batch_size):
        batch = data.iloc[i:i + batch_size]
        inputs = tokenizer(
            batch["OCR Text"].tolist(),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=512,
                num_beams=4,
                early_stopping=True
            )
            
        batch_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        for pred, truth in zip(batch_preds, batch["Ground Truth"].tolist()):
            cer = calculate_cer(pred, truth)
            preds.append(pred)
            cer_values.append(cer)
    
    results_df = pd.DataFrame({
        "OCR Text": data["OCR Text"],
        "Ground Truth": data["Ground Truth"],
        "Model Prediction": preds,
        "CER": cer_values
    })
    
    results_df.to_csv(output_file, index=False)
    return sum(cer_values) / len(cer_values)

def load_config(file):
    with open(file, "r") as f:
        config = yaml.safe_load(f)
    
    # Get BART config
    bart_config = config.get("bart", {})
    
    # Convert numerical values to proper types
    if "learning_rate" in bart_config:
        bart_config["learning_rate"] = float(bart_config["learning_rate"])
    if "num_train_epochs" in bart_config:
        bart_config["num_train_epochs"] = int(bart_config["num_train_epochs"])
    if "per_device_train_batch_size" in bart_config:
        bart_config["per_device_train_batch_size"] = int(bart_config["per_device_train_batch_size"])
    if "per_device_eval_batch_size" in bart_config:
        bart_config["per_device_eval_batch_size"] = int(bart_config["per_device_eval_batch_size"])
    if "gradient_accumulation_steps" in bart_config:
        bart_config["gradient_accumulation_steps"] = int(bart_config["gradient_accumulation_steps"])
    if "warmup_steps" in bart_config:
        bart_config["warmup_steps"] = int(bart_config["warmup_steps"])
    if "weight_decay" in bart_config:
        bart_config["weight_decay"] = float(bart_config["weight_decay"])
    if "logging_steps" in bart_config:
        bart_config["logging_steps"] = int(bart_config["logging_steps"])
    if "eval_steps" in bart_config:
        bart_config["eval_steps"] = int(bart_config["eval_steps"])
    if "save_steps" in bart_config:
        bart_config["save_steps"] = int(bart_config["save_steps"])
    if "save_total_limit" in bart_config:
        bart_config["save_total_limit"] = int(bart_config["save_total_limit"])
    
    return bart_config


def prepare_dataset(data, tokenizer, max_length=512):
    def preprocess_function(examples):
        model_inputs = tokenizer(
            examples["OCR Text"],
            max_length=max_length,
            padding=False,
            truncation=True
        )
        
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
    # Load configuration
    config = load_config(args.config)
    model_name = args.model
    output_dir = os.path.join("model", f"bart-ocr-{args.tuning}")
    
    # Load tokenizer
    tokenizer = BartTokenizer.from_pretrained(model_name)
    
    # Load data and split into train/eval
    train_df = pd.read_csv(args.data)
    # Shuffle the data
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split into train and eval (90% train, 10% eval)
    split_idx = int(len(train_df) * 0.9)
    train_df_split = train_df[:split_idx]
    eval_df_split = train_df[split_idx:]
    
    # Convert to datasets
    train_dataset = Dataset.from_pandas(train_df_split)
    eval_dataset = Dataset.from_pandas(eval_df_split)
    
    # Default arguments
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
        "num_train_epochs": 3
    }
    
    # Update defaults with config values
    training_args_dict = {**default_args, **config}
    
    # Create training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        **training_args_dict
    )
    
    if args.tuning == "eval":
        model = BartForConditionalGeneration.from_pretrained(model_name).to(device)
        eval_cer = evaluate(tokenizer, model, train_df, "./bart_evaluation_results.csv")
        print(f"Evaluation CER: {eval_cer:.4f}")
        return
    
    # Prepare model
    model = BartForConditionalGeneration.from_pretrained(model_name)
    
    if args.tuning == "full":
        # Prepare datasets
        tokenized_train_dataset = prepare_dataset(train_dataset, tokenizer)
        tokenized_eval_dataset = prepare_dataset(eval_dataset, tokenizer)
        
        # Initialize trainer
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_eval_dataset,  # Add evaluation dataset
            tokenizer=tokenizer,
            data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
        )
        
    elif args.tuning == "adapter":
        # Add adapter layers
        model.add_adapter("ocr_correction")
        model.train_adapter("ocr_correction")
        
        tokenized_train_dataset = prepare_dataset(train_dataset, tokenizer)
        tokenized_eval_dataset = prepare_dataset(eval_dataset, tokenizer)
        
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_eval_dataset,  # Add evaluation dataset
            tokenizer=tokenizer,
            data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
        )
    
    # Train and save
    trainer.train()
    trainer.save_model(output_dir)
    
    # Evaluate
    eval_cer = evaluate(tokenizer, model, train_df, "./bart_training_results.csv")
    print(f"Training CER: {eval_cer:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning BART for OCR correction")
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
        choices=["full", "adapter", "eval"],
        default="full",
        help="Specify tuning type"
    )
    
    args = parser.parse_args()
    main(args)