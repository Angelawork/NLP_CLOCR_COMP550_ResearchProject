from datasets import Dataset
from peft import AutoPeftModelForCausalLM
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig
import Levenshtein
import pandas as pd
import torch
from main import calculate_cer

def get_results(data, preds):
    df = data.to_pandas()
    df["Model Correction"] = preds
    df = df.rename(columns={"CER": "old_CER"})
    df["new_CER"] = df.apply(lambda x: calculate_cer(x["Model Correction"], x["Ground Truth"]), axis=1)
    df["CER_reduction"] = ((df["old_CER"] - df["new_CER"]) / df["old_CER"]) * 100
    return df

def test_prompt_template(example):
    return f"""### Instruction:
        Fix the OCR errors in the provided text.

        ### Input:
        {example}

        ### Response:
        """

weight_dir = "./model/llama-2-7b-ocr"
test = pd.read_csv("./dataset/test.csv")
test = Dataset.from_pandas(test)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoPeftModelForCausalLM.from_pretrained(
    weight_dir,
    quantization_config=bnb_config,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(weight_dir)

preds = []

for i in tqdm(range(len(test))):
    prompt = test_prompt_template(test[i]["OCR Text"])
    input_ids = tokenizer(prompt, max_length=1024, return_tensors="pt", truncation=True).input_ids.cuda()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids, 
            max_new_tokens=1024
        )
    pred = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):].strip()
    preds.append(pred)

results = get_results(test, preds)
results.to_csv("./llama-2-7b.csv", index=False)
