from datasets import load_dataset, DatasetDict
from huggingface_hub import login

login() 

print("Loading the MultiNLI dataset...")
try:
    full_dataset_dict = load_dataset("nyu-mll/multi_nli")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

print("Filtering for 'train' and 'validation_matched' splits...")
target_splits = ['train', 'validation_matched']
dataset_to_process = DatasetDict({
    split: full_dataset_dict[split]
    for split in target_splits
    if split in full_dataset_dict
})

def format_and_combine_text(example):
    """
    Combines 'premise' and 'hypothesis' into a single 'text' column 
    and keeps 'pairID' and 'label'.
    """
    example['text'] = f"Premise: {example['premise']} Hypothesis: {example['hypothesis']}"
    return example

columns_to_remove = [
    'promptID', 
    'premise', 
    'premise_binary_parse', 
    'premise_parse', 
    'hypothesis', 
    'hypothesis_binary_parse', 
    'hypothesis_parse', 
    'genre'
]

print("Applying transformation and removing unnecessary columns...")
modified_dataset_dict = dataset_to_process.map(
    format_and_combine_text,
    remove_columns=columns_to_remove
)

print("\n--- Modified Dataset Structure ---")
print(modified_dataset_dict)
print("\n--- Example of a Processed Row (Train Split) ---")
print(modified_dataset_dict['train'][0]) 

repo_id = "Revesis/multi_nli_setfit_formatted" 

try:
    print(f"\nUploading modified dataset to: {repo_id}")
    modified_dataset_dict.push_to_hub(repo_id)
    print(f"\nSuccessfully uploaded dataset to: https://huggingface.co/datasets/{repo_id}")
except Exception as e:
    print(f"\nError during push_to_hub. Check your login status and repo_id format. Error: {e}")