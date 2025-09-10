import argparse
import os
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Model as AzureMLModel

def main(model_path, model_name, registration_status):
    print("AZUREML_ARM_SUBSCRIPTION:", os.environ.get("AZUREML_ARM_SUBSCRIPTION"))
    print("AZUREML_ARM_RESOURCEGROUP:", os.environ.get("AZUREML_ARM_RESOURCEGROUP"))
    print("AZUREML_ARM_WORKSPACE_NAME:", os.environ.get("AZUREML_ARM_WORKSPACE_NAME"))

    try:
        credential = DefaultAzureCredential()
        ml_client = MLClient(
            credential=credential,
            subscription_id=os.environ.get("AZUREML_ARM_SUBSCRIPTION"),
            resource_group_name=os.environ.get("AZUREML_ARM_RESOURCEGROUP"),
            workspace_name=os.environ.get("AZUREML_ARM_WORKSPACE_NAME"),
        )
        model = AzureMLModel(
            name=model_name,
            path=model_path,
            description="Registered model using MLClient"
        )
        ml_client.models.create_or_update(model)
        print(f"Model {model_name} registered successfully.")
        
        # Write dummy status file for pipeline dependency
        with open(registration_status, "w") as f:
            f.write("success")
    except Exception as e:
        print("Authentication or registration failed:", e)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help="Path to the trained model")
    parser.add_argument('--model_name', type=str, help="Name for the registered model")
    parser.add_argument('--registration_status', type=str, help="Path to status file for pipeline dependency")
    args = parser.parse_args()
    main(args.model_path, args.model_name, args.registration_status)