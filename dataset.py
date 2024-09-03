import torch
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, tokenizer, source_len, target_len, source_text, target_text):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.target_len = target_len
        self.source_text = source_text
        self.target_text = target_text

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        source_text = str(self.data[self.source_text].iloc[index]).strip()
        target_text = str(self.data[self.target_text].iloc[index]).strip()
        if not source_text:
            source_text = "[EMPTY]"
        if not target_text:
            target_text = "[EMPTY]"
        # Tokenize the source text    
        source = self.tokenizer(source_text, max_length=self.source_len, padding='max_length', truncation=True, return_tensors="pt")
        # Tokenize the target text
        target = self.tokenizer(target_text, max_length=self.target_len, padding='max_length', truncation=True, return_tensors="pt")
        # Extract the input IDs and attention masks 
        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        # Clone the target IDs to use as labels, with padding tokens ignored in the loss computation
        labels = target_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100  # Ignore padding tokens in the loss computation
        return {"input_ids": source_ids, "attention_mask": source_mask, "labels": labels}
