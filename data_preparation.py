import pandas as pd
import json
import re

def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def create_dataframe(data):
    data_list = [{'id': key, 'original': value.get('original'), 'disfluent': value.get('disfluent')} for key, value in data.items()]
    return pd.DataFrame(data_list)

def clean_text(text):
    text = text.encode('ascii', 'ignore').decode('ascii')
    # Remove specific Unicode patterns
    text = re.sub(r'\\u[0-9A-Fa-f]{4}', '', text)
    # Remove non-alphanumeric characters (except spaces and punctuation)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    return text

def remove_problematic_rows(df, id_to_remove='5a6ce1054eec6b001a80a665'):
    """
    Identifies and removes rows where 'disfluent' or 'original' text fields are empty,
    and optionally removes rows based on a specific 'id'.
    
    Args:
    df (pd.DataFrame): DataFrame containing the columns 'disfluent', 'original', and 'id'.
    id_to_remove (str, optional): The specific 'id' of a row to remove. Default is '5a6ce1054eec6b001a80a665'.

    Returns:
    pd.DataFrame: DataFrame after removing rows with empty 'disfluent' or 'original' fields and the specified 'id'.
    """
    # Identify problematic rows
    problematic_rows = df[(df['disfluent'].str.strip() == '') | (df['original'].str.strip() == '')]
    print("Problematic Rows:")
    print(problematic_rows)

    # Remove problematic rows from the DataFrame
    cleaned_df = df[(df['disfluent'].str.strip() != '') & (df['original'].str.strip() != '')]

    # Additionally, remove a specific row by 'id' if provided
    if id_to_remove:
        cleaned_df = cleaned_df[cleaned_df['id'] != id_to_remove]
        print(f"Row with id={id_to_remove} has been removed.")

    return cleaned_df

def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F700-\U0001F77F"  # alchemical symbols
        u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        u"\U0001FA00-\U0001FA6F"  # Chess Symbols
        u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        u"\U00002702-\U000027B0"  # Dingbats
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_unwanted_disfluencies(df, column_name):
    """
    Removes rows from a DataFrame where the specified column's value matches any of a predefined list of unwanted values.

    Args:
    df (pd.DataFrame): The DataFrame to process.
    column_name (str): The name of the column to check for unwanted values.

    Returns:
    pd.DataFrame: A DataFrame with the unwanted values removed from the specified column.
    """
    # Predefined list of unwanted values
    values_to_remove = [
        "Question na", "question na", "no question", "No question", "No question.",
        "Question NA", "na", "no question na", "VALUE!"
    ]

    if column_name not in df.columns:
        raise ValueError(f"The column '{column_name}' does not exist in the DataFrame.")

    # Filtering out the rows where the column value is in the values_to_remove list
    filtered_df = df[~df[column_name].isin(values_to_remove)]
    return filtered_df


def preprocess_data_training(df):
   df['disfluent'] = df['disfluent'].apply(remove_emojis)
   df['disfluent'] = df['disfluent'].apply(clean_text)
   df = remove_problematic_rows(df)
   df = remove_unwanted_disfluencies(df, 'disfluent')
   return df
