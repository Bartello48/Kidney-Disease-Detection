- .env file variables
  * DATABASE_PATH="" - a file path to the training dataset
- run: uv run python -m pipelines.<pipeline_file_name>

# DATA PREPROCESSING
run from /src/kidney-disease-detection/.:
uv run python3 -m pipelines.preprocess_data -200 300 256 20
produces a folder slices and populates it with processed ct slices named:
case_{case_id:>05}_slice_{slice_number}.pt
in a directory specified in PREPROCESSED_PATH variable in .env file

saved files conatin:
image as torch.tensor [1, 256, 256]
mask as torch tensor [256, 256],
case_id as int
slice_id as int
file_path as str

can be imported to use
- get_crop
- normalize_hu
- resize
- process_case
functions