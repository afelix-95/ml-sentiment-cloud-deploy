import argparse
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Model as AzureMLModel

def main(model_path, model_name):
    try:
        credential = DefaultAzureCredential()
        ml_client = MLClient.from_config(credential=credential)
        model = AzureMLModel(
            name=model_name,
            path=model_path,
            description="Registered model using MLClient"
        )
        ml_client.models.create_or_update(model)
        print(f"Model {model_name} registered successfully.")
    except Exception as e:
        print("Authentication or registration failed:", e)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help="Path to the trained model")
    parser.add_argument('--model_name', type=str, help="Name for the registered model")
    args = parser.parse_args()
    main(args.model_path, args.model_name)