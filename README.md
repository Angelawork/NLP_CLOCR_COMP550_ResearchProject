# Exploring Tuning Strategies for Post-OCR Correction: A Comparative Analysis of Modern LLMs

Historical newspapers, with their invaluable insights into past societies and events, present unique challenges for Optical Character Recognition systems due to degraded paper quality, irregular layouts, and antiquated typography. The experiment primarily focused on evaluating the effectiveness of parameter-efficient fine-tuning with LoRA compared to full fine-tuning for open-source models, highlighting their trade-offs in error reduction and cost-efficiency. Prompt-tuning, particularly with few-shot learning, demonstrated strong performance for commercial LLMs. While commercial models outperformed open-source alternatives in terms of low error rates, the gap was modest, with fine-tuned open-source models still offering a highly scalable and relatively reliable alternative. These findings provide actionable insights into the trade-offs between fine-tuning and prompt-tuning strategies, contributing to developing efficient and scalable post-OCR correction pipelines for historical document processing. We release our synthetic dataset to facilitate further evaluation and research.

---
Our work aims to systematically review and explore the relative effectiveness of fine-tuning and prompt-tuning approaches for post-OCR correction, focusing on both open-source and commercial LLMs. Given the complexity of OCR errors in historical texts and the diverse capabilities of modern LLMs, this study adopts an exploratory perspective, recognizing model selection, evaluation metrics (CER, WER), and subjective interpretation of results. By combining designed experimentation with grounded evaluations, we aim to seek answers for:
- Empirical analysis:
  - Under what conditions do fine-tuning and prompt-tuning provide the most effective performance for post-OCR correction tasks, based on the measurement metrics (WER, CER)?
- Sensitivity to data augmentation technique:
  - Does data augmentation improve correction performance, and how sensitive is this improvement to variations in tuning strategies?
- Real-world application:
  - How do fine-tuned open-source models compare to commercial LLM solutions in terms of cost-effectiveness and viability for large-scale historical document post-correction?

## Performance of Open-Source Model trained with/without synthetic data. 
Models trained on BLN600 only are labelled "original," while those trained on augmented datasets are labelled by their target CER-WER pairs.
| **Model**                | **Augmented Dataset Used(CER-WER)** | **Train CER** | **Val CER** | **Test CER** | **Train WER** | **Val WER** | **Test WER** |
|--------------------------|------------------|---------------|-------------|--------------|---------------|-------------|--------------|
| **BART-Base: Lora**       | 0.05-0.25        | **0.0872**    | 0.0830      | 0.0845       | 0.1816        | 0.1850      | **0.1743**   |
|                          | 0.1-0.3          | 0.0974        | 0.0831      | **0.0851**   | 0.1963        | **0.1833**  | 0.1744       |
|                          | 0.15-0.4         | 0.1074        | 0.0826      | 0.0846       | 0.2116        | 0.1847      | **0.1754**   |
|                          | 0.15-0.55        | 0.1113        | **0.0823**  | **0.0832**   | 0.2202        | 0.1856      | 0.1751       |
|                          | original         | 0.0872        | 0.0832      | 0.0842       | **0.1861**    | 0.1845      | 0.1753       |
| **LLaMA-2-7B: Lora**      | 0.05-0.25        | 0.0998        | 0.0792      | **0.0921**   | 0.1204        | 0.1113      | **0.1915**   |
|                          | 0.1-0.3          | 0.0648        | 0.0633      | 0.0659       | 0.1281        | 0.1198      | 0.1113       |
|                          | 0.15-0.4         | **0.0501**    | **0.0525**  | **0.0522**   | 0.1132        | 0.1168      | 0.1212       |
|                          | 0.15-0.55        | 0.0511        | 0.0537      | 0.0539       | 0.0998        | **0.0988**  | **0.0942**   |
|                          | original         | **0.0509**    | 0.0536      | 0.0533       | **0.0932**    | 0.0990      | 0.0949       |
| **BART-Base: Full**       | 0.05-0.25        | 0.0691        | 0.0750      | **0.0764**   | 0.1258        | 0.1417      | **0.1386**   |
|                          | 0.1-0.3          | 0.0848        | **0.0730**  | 0.0734       | 0.1510        | 0.1412      | 0.1361       |
|                          | 0.15-0.4         | 0.0822        | 0.0733      | 0.0743       | 0.1462        | 0.1429      | **0.1354**   |
|                          | 0.15-0.55        | 0.0848        | **0.0730**  | **0.0733**   | 0.1510        | 0.1412      | 0.1361       |
|                          | original         | **0.0671**    | 0.0738      | 0.0752       | **0.1249**    | **0.1403**  | 0.1355       |
| **LLaMA-2-7B: Full**      | 0.05-0.25        | 0.0704        | 0.0650      | **0.0694**   | **0.0106**    | 0.1024      | 0.1130       |
|                          | 0.1-0.3          | **0.0430**    | **0.0424**  | **0.0419**   | 0.0936        | 0.0977      | **0.0862**   |
|                          | 0.15-0.4         | 0.0494        | 0.0509      | 0.0517       | 0.1035        | 0.1021      | **0.1197**   |
|                          | 0.15-0.55        | 0.0501        | 0.0520      | 0.0505       | 0.0749        | **0.0893**  | 0.0881       |
|                          | original         | 0.0539        | 0.0684      | 0.0663       | 0.0890        | 0.0932      | 0.1011       |

---

## Performance of Commercial Models Based on Prompt Tuning Strategy

| **Model**            | **Prompt Type**      | **Test CER**    | **Test WER**    |
|----------------------|----------------------|-----------------|-----------------|
| **Gemini-flash-1.5-8b** | Basic Zero-shot     | *0.0908*  | *0.3512* |
|                      | Detailed Zero-shot   | 0.0940          | 0.3335          |
|                      | One-shot             | 0.0857          | 0.2923          |
|                      | Few-shot             | **0.0753**  | **0.2577**  |
| **GPT-4o-Mini**      | Basic Zero-shot      | *0.0230*  | *0.1242*   |
|                      | Detailed Zero-shot   | 0.0217          | 0.1115          |
|                      | One-shot             | **0.0132** | 0.0937          |
|                      | Few-shot             | 0.0172          | **0.0764**  |

