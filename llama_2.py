from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from peft import PrefixTuningConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import numpy as np

from trl import SFTConfig, SFTTrainer
import argparse
import os
import pandas as pd
import torch
import yaml
from main import calculate_cer
from Levenshtein import distance as levenshtein_distance
# need to login to access model
from huggingface_hub import login
# from model_eval import test_prompt_template
hf_api_key = os.getenv("HF_API_KEY")
login(token=hf_api_key)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

def test_prompt_template(example):
    return f"""### Instruction:
        Fix the OCR errors in the provided text.

        ### Input:
        {example}

        ### Response:
        """


def evaluate(tokenizer, model, data, output_file, batch_size=1):
    preds, cer_values = [], []
    input_texts = [test_prompt_template(r) for r in data.to_dict(orient="records")]
    for i in range(0, 3, batch_size):
        batch_texts = input_texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, max_length=1024, padding=True).to(device)
        with torch.no_grad():
            outputs = model.generate(inputs.input_ids, max_new_tokens=512,pad_token_id=tokenizer.pad_token_id)
        batch_pred = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        batch_pred = [pred.strip() for pred in batch_pred]

        for pred, r in zip(batch_pred, data.iloc[i:i + batch_size].to_dict(orient="records")):
            cer = calculate_cer(pred, r["Ground Truth"])
            cer_values.append(cer)
        preds.extend(batch_pred)

    data["Model Prediction"], data["CER"] = preds, cer_values
    data.to_csv(output_file, index=False)
    return sum(cer_values) / len(cer_values)

def load_config(file):
    with open(file, "r") as f:
        config = yaml.safe_load(f)
    return config["llama-2"]

def train_prompt_template(example):
    return f"""### Instruction:
Fix the OCR errors in the provided text.

### Input:
{example["OCR Text"]}

### Response:
{example["Ground Truth"]}
"""

def main(args):
    config = load_config(args.config)
    model_name = f"meta-llama/{args.model.capitalize()}-hf"
    output_dir = os.path.join("model", f"{args.model}-ocr")

    train = pd.read_csv(args.data)
    train = Dataset.from_pandas(train)

    if torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        bnb_config = None
    config["learning_rate"] = float(config["learning_rate"])
    train_args = SFTConfig(
        output_dir=output_dir,
        **config,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if args.tuning == "eval":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config if bnb_config else None,
            use_cache=True,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        with torch.no_grad():
            train_cer = evaluate(tokenizer, model, pd.read_csv(args.data), "./evaluation_results.csv")
    else:
        if args.tuning == "full":
            train = pd.read_csv(args.data)
            train["text"] = train["OCR Text"]
            train["labels"] = train["Ground Truth"]
            train = Dataset.from_pandas(train)

            train = train.map(
                lambda x: tokenizer(
                    x["text"], 
                    padding="max_length", 
                    truncation=True, 
                    max_length=1024
                ),
                batched=True
            )

            train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

            model = AutoModelForCausalLM.from_pretrained(model_name)
            for param in model.parameters():
                param.requires_grad = True

            data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)
            config["learning_rate"] = float(config["learning_rate"])
            train_args = Seq2SeqTrainingArguments(
                output_dir=output_dir,
                **config,
            )
            trainer = Seq2SeqTrainer(
                model=model,
                args=train_args,
                train_dataset=train,
                tokenizer=tokenizer,
                data_collator=data_collator,
            )
        elif args.tuning == "lora":
            peft_config = LoraConfig(
                lora_alpha=16, #scaling factor for LoRA layers
                lora_dropout=0.1,
                r=64, #number of trainable parameters
                bias="none", # none: bias not trained, all: All biases in the model are fine-tuned, lora_only: Only biases in layers where LoRA is applied are fine-tuned
                task_type="CAUSAL_LM", #task type
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config if bnb_config else None,
                use_cache=False,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            if bnb_config:  
                model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, peft_config)

            trainer = SFTTrainer(
                model=model,
                args=train_args,
                train_dataset=train,
                peft_config=peft_config,
                max_seq_length=1024,
                tokenizer=tokenizer,
                packing=True,
                formatting_func=train_prompt_template,
            )
        elif args.tuning == "prefix":
            # micheal's part
            # peft_config = PrefixTuningConfig(
            #     prefix_dim=64,
            #     task_type="CAUSAL_LM",
            # )
            # model = AutoModelForCausalLM.from_pretrained(
            #     model_name,
            #     quantization_config=bnb_config if bnb_config else None,
            #     use_cache=False,
            #     device_map="auto" if torch.cuda.is_available() else None,
            # )
            # model = get_peft_model(model, peft_config)
            pass
        elif args.tuning == "ptuning":
            pass
        else:
            raise ValueError("Invalid tuning type specified!")
        if args.tuning != "full":
            model.to(device)
        trainer.train()
        trainer.save_model(output_dir)

    # train_cer = evaluate(tokenizer, model, pd.read_csv(args.data), "./evaluation_results.csv")
    # print(f"Overall Training CER: {train_cer:.4f}")
    # test_data = pd.read_csv("./dataset/test.csv")
    # test_cer = evaluate(tokenizer, model, test_data, "./test_results.csv")
    # print(f"Overall Test CER: {test_cer:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning for Llama")
    parser.add_argument("--model", type=str, choices=["llama-2-7b", "llama-2-13b", "llama-2-70b"],
                        default="llama-2-7b", help="Specify model: llama-2-7b, llama-2-13b, llama-2-70b")
    parser.add_argument("--config", type=str, help="Path to config", default="./config.yaml")
    parser.add_argument("--data", type=str, help="Path to training data", default="./dataset/train.csv")
    parser.add_argument("--tuning", type=str, choices=["full", "lora", "prefix", "ptuning", "eval"],
                        default="full", help="Specify tuning type")
    args = parser.parse_args()

    main(args)