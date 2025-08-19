import argparse
import os
from azureml.core import Workspace, Model
from azure.identity import DefaultAzureCredential

def main(model_path, model_name):
    # Fetch workspace details from environment variables
    subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID")
    resource_group = os.environ.get("AZURE_RESOURCE_GROUP")
    workspace_name = os.environ.get("AZURE_WORKSPACE_NAME")

    if not all([subscription_id, resource_group, workspace_name]):
        raise ValueError("Missing required environment variables: AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, or AZURE_WORKSPACE_NAME")

    # Authenticate using DefaultAzureCredential (leverages GitHub Actions credentials)
    ws = Workspace(
        subscription_id=subscription_id,
        resource_group=resource_group,
        workspace_name=workspace_name,
        auth=DefaultAzureCredential()
    )

    # Register the model
    Model.register(workspace=ws, model_path=model_path, model_name=model_name)
    print(f"Model {model_name} registered successfully in workspace {workspace_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help="Path to the trained model")
    parser.add_argument('--model_name', type=str, help="Name for the registered model")
    args = parser.parse_args()
    main(args.model_path, args.model_name)