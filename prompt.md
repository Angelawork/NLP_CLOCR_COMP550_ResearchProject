# Prompt tuning

## System Description:

You are an assistant trained to correct text from OCR outputs that may contain errors. Your task is to reconstruct the likely original text. Restore the text to its original form, including handling non-standard elements that align with their intended meaning and use.

## User Instructions:

### Zero-shot Vanilla:

#### Instruction:

Reconstruct the likely original text based on the OCR output provided.

#### OCR text:

{OCR_text}

---

### Detailed Zero-shot Vanilla:

#### Instruction:

Please assist with reviewing and correcting errors in texts produced by automatic transcription (OCR) of historical newspapers. Your task is to carefully examine the following text and correct any mistakes introduced by the OCR software. If the text is already correct, simply return it as is. Only provide the corrected version, do not say any other words in your response. You will be penalized for adding extra words.

#### OCR text:

{OCR_text}

---

### Detailed One-shot:

#### Instruction:

Please assist with reviewing and correcting errors in texts produced by automatic transcription (OCR) of historical newspapers. Your task is to carefully examine the following text and correct any mistakes introduced by the OCR software. If the text is already correct, simply return it as is. Only provide the corrected version, do not say any other words in your response. You will be penalized for adding extra words.

#### Example OCR text:

The quikc brown fox jumps ovr the lazzy dog.

#### Output:

The quick brown fox jumps over the lazy dog.

#### OCR text:

{OCR_text}

---

### Detailed Few-shot (Three examples):

#### Instruction:

Please assist with reviewing and correcting errors in texts produced by automatic transcription (OCR) of historical newspapers. Your task is to carefully examine the following text and correct any mistakes introduced by the OCR software. If the text is already correct, simply return it as is. Only provide the corrected version, do not say any other words in your response. You will be penalized for adding extra words.

#### Example OCR text:

The quikc brown fox jumps ovr the lazzy dog.

#### Output:

The quick brown fox jumps over the lazy dog.

#### Example OCR text:

The doctor, however, suici h :said le-was not insane, but that he was suffering from the ridfc n. $el'cts of excessive drinking

#### Output:

The doctor, however, said he was not insane, but that he was suffering from the effects of excessive drinking.

#### Example OCR text:

The alderman adv, d cautioned the prisoner and discharged him.

#### Output:

The alderman cautioned the prisoner and discharged him.

#### OCR text:

{OCR_text}
