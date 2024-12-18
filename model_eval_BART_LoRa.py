from transformers import BartTokenizer, BartForConditionalGeneration
import pandas as pd
import torch
from datasets import Dataset
from main import calculate_cer, calculate_wer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def evaluate_model(model_path, base_model_name, train_path, eval_path, test_path):
    tokenizer = BartTokenizer.from_pretrained(base_model_name)
    model = BartForConditionalGeneration.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    # Function to evaluate a single dataset
    def evaluate_dataset(data_path, dataset_name):
        df = pd.read_csv(data_path)
        predictions = []
        cer_values = []
        wer_values = []
        batch_size = 4
        
        print(f"\nEvaluating {dataset_name} dataset...")
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i + batch_size]
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
        
        avg_cer = sum(cer_values) / len(cer_values)
        avg_wer = sum(wer_values) / len(wer_values)
        print(f"{dataset_name} Average CER: {avg_cer:.4f}")
        print(f"{dataset_name} Average WER: {avg_wer:.4f}")
        
        results_df = pd.DataFrame({
            "OCR Text": df["OCR Text"],
            "Ground Truth": df["Ground Truth"],
            "Model Prediction": predictions,
            "CER": cer_values,
            "WER": wer_values  
        })
        
        results_df.to_csv(f"bart-Lora-{dataset_name.lower()}_results.csv", index=False)
        return avg_cer, avg_wer, results_df
    
    # Evaluate all datasets
    train_metrics = evaluate_dataset(train_path, "aggregated_wer0.55_cer0.15_train")
    eval_metrics = evaluate_dataset(eval_path, "original_Validation")
    test_metrics = evaluate_dataset(test_path, "original_Test")
    print("\nSummary of Results:")

    print("-" * 50)
    print(f"Training Set   - CER: {train_metrics[0]:.4f}, WER: {train_metrics[1]:.4f}")
    print(f"Validation Set - CER: {eval_metrics[0]:.4f}, WER: {eval_metrics[1]:.4f}")
    print(f"Test Set       - CER: {test_metrics[0]:.4f}, WER: {test_metrics[1]:.4f}")
    
    return {
        'train': train_metrics,
        'eval': eval_metrics,
        'test': test_metrics
    }

if __name__ == "__main__":
    model_path = "./model/bart-Lora-aggregated_wer0.55_cer0.15_train"
    base_model_name = "facebook/bart-base"
    
    train_path = "./dataset/aggregated_wer0.55_cer0.15_train.csv"
    eval_path = "./dataset/original_val.csv"
    test_path = "./dataset/original_test.csv"
    
    results = evaluate_model(model_path, base_model_name, train_path, eval_path, test_path)