- .env file variables
  * DATABASE_PATH="" - a file path to the training dataset
- run: uv run python -m pipelines.<pipeline_file_name>
- preprocess data: uv run python3 -m pipelines.preprocess_data -200 300 256 20