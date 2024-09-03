from data_preparation import load_data, create_dataframe, preprocess_data_training
from model_training import train_model
from dataset import TextDataset
from model_evaluating import  predict_text, calculate_metrics
from config import SOURCE_LENGTH, TARGET_LENGTH, prepare_device
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.model_selection import train_test_split
import torch
import argparse

def main():
    # Load and prepare data
    tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-base')
    model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-base')
    # train_data = load_data('/home/sahar/Downloads/query_rewriting/train.json')
    # train_df = create_dataframe(train_data)
    # train_df = preprocess_data_training(train_df)
    # train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)
    # results = train_model(train_df, val_df, source_len=262, target_len=189)
    # print(results)

    parser = argparse.ArgumentParser(description="Process the path to the test data file.")
    parser.add_argument('data_path', type=str, help="The path to the test data file.")
    args = parser.parse_args()
    # loading the test data
    test_data = load_data(args.data_path)
    # test_data = load_data('/home/sahar/Downloads/query_rewriting/dev.json')
    test_df = create_dataframe(test_data)
    # load saved model
    # model.load_state_dict(torch.load('model_weights.pth'))

    # model_save_path = './finetuned_model'
    # tokenizer_save_path = './finetuned_tokenizer'
    # model = T5ForConditionalGeneration.from_pretrained(model_save_path)
    # tokenizer = T5Tokenizer.from_pretrained(tokenizer_save_path) 
    # test the dataset
    print("Loading the pre-trained T5 model...")
    model = T5ForConditionalGeneration.from_pretrained("./saved_model")
    print("Model loaded successfully.")
    print("Loading the tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained("./saved_model")
    print("Tokenizer loaded successfully.")

    source_len = SOURCE_LENGTH
    target_len = TARGET_LENGTH
    model.eval()
    test_dataset = TextDataset(test_df, tokenizer, source_len, target_len, "disfluent", "original")
    device = prepare_device()
    print(f"Using device: {device}")
    print("Generating predictions from the model...")
    predictions, references = predict_text(model, tokenizer, test_dataset, device)
    print("Calculating and displaying metrics...")
    calculate_metrics(predictions, references,test_df)    
        


if __name__ == "__main__":
    main()