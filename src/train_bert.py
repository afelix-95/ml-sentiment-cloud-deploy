import argparse
import os
from azureml.core import Workspace, Model
from azure.identity import DefaultAzureCredential
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(-1)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions)
    }

def main(input_data, model_output):
    # Connect to Azure ML workspace
    ws = Workspace(
        subscription_id=os.environ.get("AZURE_SUBSCRIPTION_ID"),
        resource_group=os.environ.get("AZURE_RESOURCE_GROUP"),
        workspace_name=os.environ.get("AZURE_WORKSPACE_NAME"),
        auth=DefaultAzureCredential()
    )

    # Check for existing model
    model_name = "bert-sentiment-model"
    try:
        model = Model(ws, name=model_name)
        model.download(target_dir=model_output)
        print(f"Loading existing model {model_name}")
        model = AutoModelForSequenceClassification.from_pretrained(model_output)
    except:
        print(f"No existing model {model_name} found, starting from pre-trained BERT")
        model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    dataset = load_from_disk(input_data)
    train_test = dataset.train_test_split(test_size=0.2)

    training_args = TrainingArguments(
        output_dir=model_output,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        report_to='azureml'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_test['train'],
        eval_dataset=train_test['test'],
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(model_output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str)
    parser.add_argument('--model_output', type=str)
    args = parser.parse_args()
    main(args.input_data, args.model_output)