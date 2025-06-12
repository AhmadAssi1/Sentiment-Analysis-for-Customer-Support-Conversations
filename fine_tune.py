import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load the pre-trained model and tokenizer
model_name = "aubmindlab/bert-base-arabertv02-twitter"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3  # For negative, neutral, positive
)

def prepare_dataset(data_path):
    # Load your dataset here
    df = pd.read_csv(data_path)
    
    # Convert text labels to numeric
    label_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    df['label'] = df['label'].map(label_map)
    
    # Rename 'expression' column to 'text' for consistency
    df = df.rename(columns={'expression': 'text'})
    
    # Convert to Dataset format
    dataset = Dataset.from_pandas(df)
    
    # Split dataset
    train_testvalid = dataset.train_test_split(test_size=0.2, seed=42)
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5, seed=42)
    
    return {
        'train': train_testvalid['train'],
        'validation': test_valid['train'],
        'test': test_valid['test']
    }

def tokenize_function(examples):
    result = tokenizer(
        examples['text'],
        padding=False,
        truncation=True,
        max_length=128
    )
    result["labels"] = examples["label"]
    return result

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Calculate accuracy
    accuracy = (predictions == labels).mean()
    
    return {
        'accuracy': accuracy
    }

def main():
    # Prepare dataset
    data_path = 'arabic_sentiment_lexicon - arabic_sentiment_lexicon (1).csv'
    datasets = prepare_dataset(data_path)
    
    # Tokenize datasets
    tokenized_datasets = {
        split: dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        for split, dataset in datasets.items()
    }
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=10,
        eval_steps=50,
        save_steps=50,
        logging_dir="./logs"
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    trainer.train()
    
    # Save the model
    import os
    import shutil
    
    # Create a temporary directory for saving
    temp_dir = "./temp_model"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    # Save the model to temporary directory
    trainer.save_model(temp_dir)
    
    # Move to final location
    final_dir = "./fine_tuned_model"
    if os.path.exists(final_dir):
        shutil.rmtree(final_dir)
    shutil.move(temp_dir, final_dir)
    
    # Evaluate on test set
    test_results = trainer.evaluate(tokenized_datasets['test'])
    print(f"Test results: {test_results}")

if __name__ == "__main__":
    main() 