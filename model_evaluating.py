import torch
import json
import pandas as pd
from datasets import load_metric
from transformers import T5Tokenizer, T5ForConditionalGeneration

def prepare_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_text(model, tokenizer, test_dataset, device):
    model.to(device)
    model.eval()  
    predictions = []
    references = []
    with torch.no_grad():
        for i in range(len(test_dataset)):
            batch = test_dataset[i]
            input_ids = batch["input_ids"].unsqueeze(0).to(device)
            attention_mask = batch["attention_mask"].unsqueeze(0).to(device)
            labels = batch["labels"].unsqueeze(0).to(device)

            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, do_sample=True, top_k=120, top_p=0.98, early_stopping=True, num_return_sequences=4)
            predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            labels = labels[labels != tokenizer.pad_token_id]
            reference_text = tokenizer.decode(labels[0], skip_special_tokens=True)

            predictions.append(predicted_text)
            references.append(reference_text)
    return predictions, references

def calculate_metrics(predictions, references):
    bleu_metric = load_metric("sacrebleu")
    rouge_metric = load_metric("rouge")
    
    bleu_score = bleu_metric.compute(predictions=predictions, references=[[ref] for ref in references])
    rouge_score = rouge_metric.compute(predictions=predictions, references=[[ref] for ref in references])

    print(f"BLEU score: {bleu_score['score']}")
    print(f"ROUGE scores: {rouge_score}")
