import pandas as pd
import scrambledtext
from scrambledtext import ProbabilityDistributions, CorruptionEngine
import genalog
from genalog.genalog.text import alignment,anchor

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
import main
from main import calculate_cer

def check_unused_char(df):
    all_text = "".join(df["OCR Text"].dropna())
    symbols_set = {char for char in all_text if not char.isalnum()}
    print(symbols_set)

    all_ascii_chars = {chr(i) for i in range(32, 256)}
    unused_chars = {c for c in all_ascii_chars - symbols_set if not c.isalnum()}
    print("Characters not in the used list:")
    print(unused_chars)
    return unused_chars
# NOTE: unused chars are {'§', '\x87', '×', '\x83', '\x9a', '\x9d', '\xa0', '·', '¤', '¥', '»', '\x8d', '\x84', '\x9c', '¯', '¿', '\x8c', '\xad', '\x89', '\x82', '\x7f', '\x81', '¬', '\x85', '\x93', '\x9e', '\x8a', '®', '\x94', '¨', '÷', '\x9b', '\x92', '\x9f', '\x8e', '\x98', '\x86', '«', '\x80', '\x99', '\x95', '¦', '\x8f', '\x91', '¡', '\x88', '¸', '\x90', '©', '\x8b', '\x96', '\x97', '´'}
    
df=pd.read_csv('./dataset/data.csv')

df["Aligned OCR Text"] = ""
df["Aligned Ground Truth"] = ""
df["Aligned CER"] = 0.0
difference_cer=[]

for i, row in df.iterrows():
    aligned_gt,  aligned_ocr = anchor.align_w_anchor(str(row["Ground Truth"]), str(row["OCR Text"]), gap_char="©")
    cer=calculate_cer(aligned_ocr,aligned_gt)
    df.at[i, "Aligned OCR Text"] =  aligned_ocr
    df.at[i, "Aligned Ground Truth"] = aligned_gt
    df.at[i, "Aligned CER"] = cer

    difference_cer.append(row["CER"]-cer)
    if i%1000==0: 
        print(f"---processing {i}th text---")

df.to_csv("./dataset/data_aligned.csv", index=False)
print(f"Max CER difference after alignment:{max(difference_cer)}")
print(f"Min CER difference after alignment:{min(difference_cer)}")
print(f"Average CER difference after alignment:{sum([abs(x) for x in difference_cer])/len(difference_cer)}")
# Max CER difference after alignment:1.263157894736842
# Min CER difference after alignment:-0.27304441349640796
# Average CER difference after alignment:0.007272683003945741

aligned_texts = list(zip(df['Aligned OCR Text'].head(14), df['Aligned Ground Truth'].head(14)))
gen_probs = ProbabilityDistributions(aligned_texts)
scrambler = CorruptionEngine(
    gen_probs.conditional,
    gen_probs.substitutions,
    gen_probs.insertions,
    target_wer=0.9,
    target_cer=0.8,
)
gen_probs.save_to_json('./dataset/engine_probabilities.json')
original_text = "This is a sample text to be corrupted.This is a sample text to be corrupted.This is a sample text to be corrupted."
corrupted_text, actual_wer, actual_cer, effective_cer = scrambler.corrupt_text(original_text)

print(f"---Testing correuption---")
print(f"Original: {original_text}")
print(f"Corrupted: {corrupted_text}")
print(f"Actual WER: {actual_wer:.2f}")
print(f"Actual CER: {actual_cer:.2f}")
print(f"Effective CER: {effective_cer:.2f}")