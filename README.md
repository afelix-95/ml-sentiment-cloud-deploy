# End-to-End ML Model Deployment with Azure ML and GitHub Actions

This project demonstrates an end-to-end machine learning pipeline for sentiment analysis using the IMDb movie reviews dataset from Kaggle. It leverages Azure Machine Learning (Azure ML) to preprocess data, fine-tune a BERT model, and register the model, with automation through GitHub Actions for CI/CD.

## Project Overview

- **Objective**: Build and deploy a sentiment analysis model to classify IMDb reviews as positive or negative.
- **Dataset**: IMDb Dataset of 50K Movie Reviews from Kaggle, containing balanced positive and negative reviews.
- **Model**: Fine-tuned BERT (`bert-base-uncased`) using Hugging Face Transformers.
- **Platform**: Azure ML for scalable training and model management.
- **Automation**: GitHub Actions for CI/CD, triggering data ingestion, preprocessing, training, and model registration.

## Repository Structure

```
├── .github/workflows/train-bert.yml    # GitHub Actions workflow for CI/CD
├── azureml-job.yml                     # Azure ML pipeline definition
├── cpu-environment.yml                 # Conda environment for CPU jobs
├── gpu-environment.yml                 # Conda environment for GPU jobs
├── src/
│   ├── preprocess.py                   # Script for data cleaning and tokenization
│   ├── train_bert.py                   # Script for BERT model training
│   └── register_model.py               # Script for model registration in Azure ML
├── README.md                           # This file
└── LICENSE                             # MIT License
```

## Prerequisites

- **Azure Subscription**: An active Azure account with an Azure ML workspace and compute cluster (CPU or GPU).
- **GitHub Repository**: Fork or clone this repository.
- **Secrets**:
  - `AZURE_CREDENTIALS`: Azure service principal JSON (Contributor role on ML workspace).
  - `AZURE_RESOURCE_GROUP`: Your resource group name.
  - `AZURE_WORKSPACE_NAME`: Your Azure ML workspace name.
  - `KAGGLE_USERNAME` and `KAGGLE_KEY`: For downloading the IMDb dataset.
- **Tools**:
  - Azure CLI with `ml` extension (`az extension add -n ml -y`).
  - Python 3.10 with dependencies (see environment files below).

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/afelix-95/ml-sentiment-cloud-deploy.git
   cd ml-sentiment-cloud-deploy
   ```

2. **Configure Azure ML**:
   - Set up an Azure ML workspace. You can create new compute resources or use serverless compute.

3. **Set Up GitHub Secrets**:
   - In your GitHub repository, go to **Settings > Secrets and variables > Actions > New repository secret**.
   - Add:
     - `AZURE_CREDENTIALS`: Service principal JSON.
     - `AZURE_RESOURCE_GROUP`: Your resource group name.
     - `AZURE_WORKSPACE_NAME`: Your Azure ML workspace name.
     - `KAGGLE_USERNAME`: Your Kaggle username.
     - `KAGGLE_KEY`: Your Kaggle API key.

## Pipeline Workflow

The pipeline is defined in `azureml-job.yml` and executed via `train-bert.yml` in GitHub Actions. Steps include:

1. **Data Ingestion**: Downloads IMDb dataset from Kaggle and uploads to Azure ML datastore.
2. **Preprocessing**: Cleans reviews, tokenizes using BERT tokenizer, and saves as a Hugging Face dataset.
3. **Training**: Fine-tunes BERT for binary classification (positive/negative) on a GPU cluster.
4. **Model Registration**: Registers the trained model in Azure ML.

### Triggering the Pipeline
- **Automatic**: Pushes to `main` or pull requests trigger the workflow.
- **Manual**: Use GitHub Actions' "Run workflow" button (`workflow_dispatch`).
- **Local Testing**:
  ```bash
  az ml job create --file azureml-job.yml --workspace-name your-ml-workspace --resource-group your-resource-group
  ```

## Scripts

- **preprocess.py**: Loads the IMDb CSV, maps sentiments to labels (positive=1, negative=0), tokenizes reviews, and saves the dataset.
- **train_bert.py**: Fine-tunes BERT with Hugging Face Trainer, logs metrics (accuracy, F1-score) to Azure ML.
- **register_model.py**: Registers the trained model in Azure ML registry.

## Dependencies

The dependencies for this project are managed using two environment files:

- **cpu-environment.yml**: For CPU-based jobs (e.g., preprocessing)
- **gpu-environment.yml**: For GPU-based jobs (e.g., model training)

**cpu-environment.yml**
```yaml
name: bert-cpu-env
channels:
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - pip:
      - azure-ai-ml
      - azure-identity
      - transformers
      - datasets
```

**gpu-environment.yml**
```yaml
name: bert-gpu-env
channels:
  - conda-forge
  - pytorch
dependencies:
  - python=3.10
  - pip
  - pip:
      - torch
      - azure-ai-ml
      - azure-identity
      - scikit-learn
      - transformers
      - accelerate
      - datasets
```

You can create and register these environments in Azure ML using:
```bash
az ml environment create --file cpu-environment.yml --workspace-name your-ml-workspace --resource-group your-resource-group
az ml environment create --file gpu-environment.yml --workspace-name your-ml-workspace --resource-group your-resource-group
```

## Monitoring and Outputs

- **Azure ML Studio**: View job status, logs, and metrics under **Jobs** > `imdb-sentiment-bert`.
- **GitHub Actions**: Check workflow logs for errors; artifacts (logs) are uploaded if configured.
- **Model Registry**: Registered model (`bert-sentiment-model`) appears in Azure ML **Models**.

## Future Enhancements

- Add model deployment to an Azure ML endpoint.
- Implement hyperparameter tuning with Azure ML AutoML.
- Integrate monitoring for data drift or performance metrics.
- Add unit tests for scripts in the GitHub workflow.

## Troubleshooting

- **Authentication Errors**: Verify `AZURE_CREDENTIALS` or run `az login` locally.
- **Dataset Issues**: Ensure the IMDb dataset is in the datastore and `dataset_path` is correct.
- **Compute Errors**: Confirm the compute cluster is running and has sufficient resources.
- **Logs**: Check `.azureml/logs/` or Azure ML Studio for detailed errors.

## License

MIT License. See [LICENSE](LICENSE) for details.
