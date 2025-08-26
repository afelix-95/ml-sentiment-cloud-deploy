import argparse
import os
import shutil
import tempfile
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
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
        subscription_id=os.environ.get("AZUREML_ARM_SUBSCRIPTION"),
        resource_group_name=os.environ.get("AZUREML_ARM_RESOURCEGROUP"),
        workspace_name=os.environ.get("AZUREML_ARM_WORKSPACE_NAME"),
    )

    # Check for existing model
    model_name = "distilbert-sentiment-model"
    try:
        model_list = ml_client.models.list(name=model_name)
        if model_list:
            model = model_list[0]
            temp_model_path = os.path.join(tempfile.gettempdir(), "downloaded_model")
            model_path = ml_client.models.download(name=model.name, version=model.version, download_path=temp_model_path)
            print(f"Loading existing model {model_name}")
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            raise Exception("Model not found")
    except Exception as e:
        print(f"No existing model {model_name} found, starting from pre-trained DistilBERT")
        model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    # Create a temporary directory for dataset
    temp_dir = tempfile.mkdtemp(dir='/tmp')

    # Copy input dataset to temporary directory
    dataset_path = os.path.join(temp_dir, 'dataset')
    shutil.copytree(input_data, dataset_path)

    # Load dataset from temporary directory
    dataset = load_from_disk(dataset_path)
    train_test = dataset.train_test_split(test_size=0.5)

    # Create a temporary directory for trainer model saving
    temp_model_dir = tempfile.mkdtemp(dir='/tmp')

    training_args = TrainingArguments(
        output_dir=temp_model_dir,  # Save checkpoints to temp directory
        num_train_epochs=1,
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

    # Save the trainer model to temp_model_dir
    trainer.save_model(temp_model_dir)
    tokenizer.save_pretrained(temp_model_dir)  # Save tokenizer

    # Clear model_output directory if it exists
    if os.path.exists(model_output):
        shutil.rmtree(model_output)
    os.makedirs(model_output)

    # Save and log the model as an MLflow artifact
    with mlflow.start_run() as run:
        mlflow.transformers.save_model(
            transformers_model={"model": model, "tokenizer": tokenizer},
            path=model_output,
            task="text-classification",
            input_example=train_test['test'][0]
        )

    # Clean up temporary directories
    shutil.rmtree(temp_dir)
    shutil.rmtree(temp_model_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str)
    parser.add_argument('--model_output', type=str)
    args = parser.parse_args()
    main(args.input_data, args.model_output)