from transformers import BartTokenizer, BartForConditionalGeneration
import pandas as pd
import torch
from datasets import Dataset
from main import calculate_cer, calculate_wer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model(model_path, base_model_name, eval_data_path):
    tokenizer = BartTokenizer.from_pretrained(base_model_name)
    
    model = BartForConditionalGeneration.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    eval_df = pd.read_csv(eval_data_path)
    
    predictions = []
    cer_values = []
    wer_values = []
    batch_size = 4
    
    for i in range(0, len(eval_df), batch_size):
        batch = eval_df.iloc[i:i + batch_size]
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
            wer = calculate_wer(pred, truth)
            predictions.append(pred)
            cer_values.append(cer)
            wer_values.append(wer)
            
            # print(f"\nOriginal: {truth}")
            # print(f"Predicted: {pred}")
            # print(f"CER: {cer:.4f}")
            # print(f"WER: {wer:.4f}")
    
    avg_cer = sum(cer_values) / len(cer_values)
    avg_wer = sum(wer_values) / len(wer_values)
    print(f"\nAverage CER: {avg_cer:.4f}")
    print(f"\nAverage WER: {avg_wer:.4f}")
    
    results_df = pd.DataFrame({
        "OCR Text": eval_df["OCR Text"],
        "Ground Truth": eval_df["Ground Truth"],
        "Model Prediction": predictions,
        "CER": cer_values,
        "WER": wer_values  
    })
    
    results_df.to_csv("bart-full-aggregated_wer0.55_cer0.15.csv", index=False)
    return avg_cer, avg_wer

if __name__ == "__main__":
    model_path = "./model/bart-full-aggregated_wer0.55_cer0.15_train"
    base_model_name = "facebook/bart-base" 
    eval_data_path = "./dataset/original_val.csv" 
    
    avg_cer, avg_wer = evaluate_model(model_path, base_model_name, eval_data_path)