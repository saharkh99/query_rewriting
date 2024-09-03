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
            # Prepare the labels, adding a batch dimension
            labels = batch["labels"].unsqueeze(0).to(device)
             # Generate predictions using the model
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, do_sample=True, top_k=120, top_p=0.98, early_stopping=True, num_return_sequences=4)
            # Decode the generated output tokens into text
            predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Filter out padding tokens from the labels
            labels = labels[labels != tokenizer.pad_token_id]
             # Decode the reference labels into text
            reference_text = tokenizer.decode(labels[0], skip_special_tokens=True)

            predictions.append(predicted_text)
            references.append(reference_text)
    return predictions, references

def calculate_metrics(predictions, references, test_df):
    bleu_metric = load_metric("sacrebleu")
    bleu_score = bleu_metric.compute(predictions=predictions[:], references=[[ref] for ref in test_df['original']])
    print(f"BLEU score: {bleu_score['score']}")

    # ROUGE score
    rouge_metric = load_metric("rouge")
    rouge_score = rouge_metric.compute(predictions=predictions[:], references=[[ref] for ref in test_df['original']])
    print("ROUGE scores:")
    print(f"ROUGE scores: {rouge_score}")
