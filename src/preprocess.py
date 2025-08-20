import argparse
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

def main(input_data, output_data):
    df = pd.read_csv(f"{input_data}/IMDB Dataset.csv")
    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    dataset = Dataset.from_pandas(df[['review', 'label']])
    
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    def tokenize(examples):
        return tokenizer(examples['review'], padding='max_length', truncation=True, max_length=256)
    
    tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=['review'])
    tokenized_dataset.save_to_disk(output_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str)
    parser.add_argument('--output_data', type=str)
    args = parser.parse_args()
    main(args.input_data, args.output_data)