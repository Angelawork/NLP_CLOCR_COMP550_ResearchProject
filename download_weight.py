from peft import AutoPeftModelForCausalLM
import torch
print(torch.cuda.is_available())

from transformers import AutoTokenizer, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoPeftModelForCausalLM.from_pretrained(
    'pykale/llama-2-13b-ocr',
    quantization_config=bnb_config,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained('pykale/llama-2-13b-ocr')

ocr = "The defendant wits'fined �5 and costs."

prompt = f"""### Instruction:
Fix the OCR errors in the provided text.

### Input:
{ocr}

### Response:
"""

input_ids = tokenizer(prompt, max_length=1024, return_tensors='pt', truncation=True).input_ids.cuda()
with torch.inference_mode():
    outputs = model.generate(input_ids=input_ids, max_new_tokens=1024, do_sample=True, temperature=0.7, top_p=0.1, top_k=40)
pred = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):].strip()

print(pred)
