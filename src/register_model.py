import argparse
import os
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Model as AzureMLModel

def main(model_path, model_name):
    # Authenticate and create MLClient
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=os.environ.get("AZURE_SUBSCRIPTION_ID"),
        resource_group=os.environ.get("AZURE_RESOURCE_GROUP"),
        workspace_name=os.environ.get("AZURE_WORKSPACE_NAME"),
    )

    # Register the model
    model = AzureMLModel(
        name=model_name,
        path=model_path,
        description="Registered model using MLClient"
    )
    ml_client.models.create_or_update(model)
    print(f"Model {model_name} registered successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help="Path to the trained model")
    parser.add_argument('--model_name', type=str, help="Name for the registered model")
    args = parser.parse_args()
    main(args.model_path, args.model_name)