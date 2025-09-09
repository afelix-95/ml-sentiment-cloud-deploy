import argparse
import os
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment

def main(model_name, endpoint_name, deployment_name):
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=os.environ.get("AZUREML_ARM_SUBSCRIPTION"),
        resource_group_name=os.environ.get("AZUREML_ARM_RESOURCEGROUP"),
        workspace_name=os.environ.get("AZUREML_ARM_WORKSPACE_NAME"),
    )

    # Create endpoint
    endpoint = ManagedOnlineEndpoint(
        name=endpoint_name,
        auth_mode="key"
    )
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()

    # Create deployment
    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=model_name,
        instance_type="Standard_DS3_v2",
        instance_count=1
    )
    ml_client.online_deployments.begin_create_or_update(deployment).result()
    print(f"Model {model_name} deployed to endpoint {endpoint_name} as deployment {deployment_name}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help="Registered model name")
    parser.add_argument('--endpoint_name', type=str, required=True, help="Online endpoint name")
    parser.add_argument('--deployment_name', type=str, required=True, help="Deployment name")
    args = parser.parse_args()
    main(args.model_name, args.endpoint_name, args.deployment_name)