from data_preparation import load_data, create_dataframe, preprocess_data_training
from model_training import train_model
from dataset import TextDataset
from model_evaluating import  predict_text, calculate_metrics
from config import SOURCE_LENGTH, TARGET_LENGTH, prepare_device

def main():
    # Load and prepare data
    # train_data = load_data('/content/train.json')
    # train_df = create_dataframe(train_data)
    # train_df = preprocess_data_training(train_df)
    # train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)
    # results = train_model(train_df, val_df, source_len=262, target_len=189)
    # print(results)


    # loading the test data
    test_data = load_data('/home/sahar/Downloads/query_rewriting/dev.json')
    test_df = create_dataframe(test_data)
    # load saved model
    

    # test the dataset
    source_len = SOURCE_LENGTH
    target_len = TARGET_LENGTH

    test_dataset = TextDataset(test_df, tokenizer, source_len, target_len, "disfluent", "original")

    device = prepare_device()
    predictions, references = predict_text(model, tokenizer, test_dataset, device)
    calculate_metrics(predictions, references)    
        


if __name__ == "__main__":
    main()