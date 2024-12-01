import json
import scrambledtext
from scrambledtext import ProbabilityDistributions, CorruptionEngine
import pandas as pd
import re

def remove_gap_char(data, char):
    """
    removes all instances of the specified char
    If a dictionary key's value becomes empty after removal, the key is also removed.
    """ 
    if isinstance(data, dict):
        for k, v in data.items():
            if k == char:
                print(f"Encountered key to remove: {k}")
            data[k] = remove_gap_char(v, char)
        return {k: v for k, v in data.items() if k != char and v != {}}
    elif isinstance(data, list):
        return [remove_gap_char(item, char) for item in data]
    else:
        return data

def split_text_by_length(text, target_length=130, max_length=1675):
    sents = re.split("(?<=\.) ", text)
    chunks = []
    chunk = ""
    for s in sents:
        if len(s) > max_length:
            continue
        if len(chunk) + len(s) + 1 <= target_length:
            chunk += s + " "
        else:
            chunks.append(chunk.strip())
            chunk = s + " "
    if chunk and len(chunk.strip()) > 0:
        chunks.append(chunk.strip())
    return chunks
   
def clean_prob():
    file_path = "./dataset/engine_probabilities.json"
    with open(file_path, "r") as file:
        data = json.load(file)

    cleaned_data = remove_gap_char(data, "©")
    with open("./dataset/engine_probabilities_cleaned2.json", "w") as file:
        json.dump(cleaned_data, file, indent=4)

def extract_samples(file):
    df = pd.read_csv(file)
    df = df.drop(columns=["Summary"])
    df = df.sample(n=50, random_state=24)

    df["Ground Truth"] = df["Content"].apply(lambda x: split_text_by_length(str(x)))
    df = df.explode("Ground Truth").drop(columns=["Content"])
    df = df.reset_index(drop=True)

    gt_lengths=[]
    for _, row in df.iterrows():
        gt_text = str(row["Ground Truth"])
        gt_lengths.append(len(gt_text))
    average_gt_length = sum(gt_lengths) / len(gt_lengths) if gt_lengths else 0
    print(len(df))
    print(f"Average Ground Truth Length: {average_gt_length:.2f} characters")
    print(f"MAX Ground Truth Length: {max(gt_lengths):.2f} characters")
    print(f"MIN Ground Truth Length: {min(gt_lengths):.2f} characters")
    df.to_csv("sample.csv")

if __name__ == "__main__":
    WER=0.55
    CER=0.17

    probs = ProbabilityDistributions.load_from_json("./dataset/synthetic_data/engine_probabilities_cleaned.json")
    engine = CorruptionEngine(
        probs.conditional,
        probs.substitutions,
        probs.insertions,
        target_wer=WER,
        target_cer=CER,
    )

    df = pd.read_csv("./dataset/synthetic_data/synthetic_origin_news.csv")
    df=df.drop(columns=["Index","random"])
    def corrupt_text(row):
        text = str(row["Ground Truth"])
        corrupted_text, actual_wer, actual_cer, effective_cer = engine.corrupt_text(text)
        return pd.Series([corrupted_text, actual_wer, actual_cer, effective_cer])
    df[["OCR Text", "Actual WER", "Actual CER", "Effective CER"]] = df.apply(corrupt_text, axis=1)
    df.to_csv(f"./dataset/synthetic_wer{WER}_cer{CER}.csv")