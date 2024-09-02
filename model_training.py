import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from dataset import TextDataset

def train_model(train_df, val_df, source_len, target_len):
    """
        Trains and evaluates a T5 model using the provided training and validation datasets.

        Args:
        train_df (pd.DataFrame): Training dataset DataFrame.
        val_df (pd.DataFrame): Validation dataset DataFrame.
        source_len (int): The maximum length of the source text.
        target_len (int): The maximum length of the target text.

        Returns:
        Dict: A dictionary containing evaluation results.
        """
    # Initialize tokenizer and model from the T5 base pretrained version
    tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-base')
    model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-base')

    # Prepare datasets for training and validation
    train_dataset = TextDataset(train_df, tokenizer, source_len, target_len, "disfluent", "original")
    val_dataset = TextDataset(val_df, tokenizer, source_len, target_len, "disfluent", "original")

    # Set training arguments
    training_args = TrainingArguments(
        output_dir='./results',  
        num_train_epochs=3,  
        per_device_train_batch_size=4,  
        per_device_eval_batch_size=4,  
        warmup_steps=500,  
        weight_decay=0.04,  
        logging_dir='./logs',  
        logging_steps=10, 
        evaluation_strategy="epoch",  
        learning_rate=6.602358273531259e-05  
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    # Train the model
    trainer.train()
    # Save 
    model.save_pretrained('./saved_model')
    # Evaluate the model
    return trainer.evaluate()

