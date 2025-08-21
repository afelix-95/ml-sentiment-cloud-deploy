import argparse
import os
import shutil
import tempfile
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_from_disk, disable_caching
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.transformers

# Disable caching to prevent writing to input directory
disable_caching()

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(-1)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions)
    }

def main(input_data, model_output):
    # Connect to Azure ML workspace using MLClient
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=os.environ.get("AZURE_SUBSCRIPTION_ID"),
        resource_group=os.environ.get("AZURE_RESOURCE_GROUP"),
        workspace_name=os.environ.get("AZURE_WORKSPACE_NAME"),
    )

    # Check for existing model
    model_name = "distilbert-sentiment-model"
    model_path = None
    try:
        model_list = ml_client.models.list(name=model_name)
        if model_list:
            model = model_list[0]
            model_path = ml_client.models.download(name=model.name, version=model.version, download_path=model_output)
            print(f"Loading existing model {model_name}")
            model = AutoModelForSequenceClassification.from_pretrained(model_output)
        else:
            raise Exception("Model not found")
    except Exception as e:
        print(f"No existing model {model_name} found, starting from pre-trained DistilBERT")
        model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

    # Create a temporary directory for dataset
    temp_dir = tempfile.mkdtemp(dir='/tmp')

    # Copy input dataset to temporary directory
    shutil.copytree(input_data, os.path.join(temp_dir, 'dataset'))
    dataset_path = os.path.join(temp_dir, 'dataset')

    # Load dataset from temporary directory
    dataset = load_from_disk(dataset_path)
    train_test = dataset.train_test_split(test_size=0.3)

    training_args = TrainingArguments(
        output_dir=model_output,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        fp16=True,  # Use mixed precision training
        # Avoid caching to prevent writes to read-only directories
        disable_tqdm=False
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

    # Log the model in MLflow format for Azure ML registration
    with mlflow.start_run() as run:
        mlflow.transformers.log_model(
            transformers_model=model,
            artifact_path="model",
            task="text-classification",
            input_example=train_test['test'][0]
        )
        mlflow_model_dir = os.path.join(mlflow.get_artifact_uri(), "model")
        print("MLflow model directory:", mlflow_model_dir)

    # Clean up temporary directory
    shutil.rmtree(temp_dir)

    if os.path.exists(model_output):
        shutil.rmtree(model_output)
    shutil.copytree(mlflow_model_dir, model_output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str)
    parser.add_argument('--model_output', type=str)
    args = parser.parse_args()
    main(args.input_data, args.model_output)